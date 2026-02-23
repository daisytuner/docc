"""Managed memory allocation and deallocation for SDFG code generation."""


class ManagedMemoryHandler:
    """
    Handles memory allocation with automatic cleanup.

    Tracks allocations whose size depends only on function arguments
    (hoistable) and emits:
    - malloc at scope start (hoisted)
    - optional memset for zero init
    - free before scope exit

    Allocations with sizes depending on loop variables are rejected
    and must be handled immediately by the caller.
    """

    # Initialization types
    INIT_NONE = "none"
    INIT_ZERO = "zero"  # Use memset(0)

    def __init__(self, builder):
        self.builder = builder
        self._allocations = []

    def allocate(self, name, ptr_type, total_size, init=INIT_NONE):
        """
        Try to register an allocation for hoisting.

        Checks if the size expression depends only on function arguments.
        If so, tracks the allocation and returns True. Otherwise returns False
        and the caller must emit the allocation immediately.

        Args:
            name: Container name for the pointer
            ptr_type: Pointer type (e.g., Pointer(Scalar(PrimitiveType.Float)))
            total_size: Size expression in bytes (string)
            init: Initialization type (INIT_NONE or INIT_ZERO)

        Returns:
            True if allocation was accepted (hoistable), False otherwise
        """
        # Check if size is hoistable (depends only on function arguments)
        if not self.builder.is_hoistable_size(total_size):
            return False

        self._allocations.append(
            {
                "name": name,
                "ptr_type": ptr_type,
                "total_size": total_size,
                "init": init,
            }
        )
        return True

    def emit_allocations(self):
        """
        Emit malloc (and memset for INIT_ZERO) at scope start.

        Must be called after parsing is complete but before builder.move().
        Allocations are emitted in reverse order since insert_block_at_root_start prepends.
        """
        for alloc in reversed(self._allocations):
            # For INIT_ZERO, insert memset first (will be after malloc due to prepending)
            if alloc["init"] == self.INIT_ZERO:
                block = self.builder.insert_block_at_root_start()
                t_memset = self.builder.add_memset(block, "0", alloc["total_size"])
                t_ptr = self.builder.add_access(block, alloc["name"])
                self.builder.add_memlet(
                    block, t_memset, "_ptr", t_ptr, "void", "", alloc["ptr_type"]
                )

            # Insert malloc (will end up before memset)
            block = self.builder.insert_block_at_root_start()
            t_malloc = self.builder.add_malloc(block, alloc["total_size"])
            t_ptr = self.builder.add_access(block, alloc["name"])
            self.builder.add_memlet(
                block, t_malloc, "_ret", t_ptr, "void", "", alloc["ptr_type"]
            )

    def emit_frees(self):
        """
        Emit free calls for all allocations.

        Should be called before each return statement.
        """
        for alloc in self._allocations:
            block = self.builder.add_block()
            t_free = self.builder.add_free(block)
            t_ptr_in = self.builder.add_access(block, alloc["name"])
            t_ptr_out = self.builder.add_access(block, alloc["name"])
            # Input memlet: access_node -> free._ptr
            self.builder.add_memlet(
                block, t_ptr_in, "void", t_free, "_ptr", "", alloc["ptr_type"]
            )
            # Output memlet: free._ptr -> access_node (required for library node)
            self.builder.add_memlet(
                block, t_free, "_ptr", t_ptr_out, "void", "", alloc["ptr_type"]
            )

    def has_allocations(self):
        """Check if there are any managed allocations."""
        return len(self._allocations) > 0

    def clear(self):
        """Clear all tracked allocations."""
        self._allocations.clear()
