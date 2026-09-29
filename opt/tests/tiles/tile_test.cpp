#include <gtest/gtest.h>

#include "sdfg/tiles/tile.h"

using namespace sdfg::tiles;

namespace {
/// The GPU scratchpad hierarchy used to build axes in these neutral-core tests.
Space gpu_space(Level level) {
    switch (level) {
        case Level::Device:
            return Space::Global;
        case Level::Group:
            return Space::Shared;
        case Level::Subgroup:
            return Space::Register;
    }
    return Space::Register;
}

TileAxis axis(Role r, Level lvl) {
    return TileAxis(sdfg::symbolic::symbol("i"), r, AxisSchedule(lvl, gpu_space(lvl), /*has_scratchpad=*/true));
}
} // namespace

TEST(TileTest, RequiredSpacePicksCoarsestCooperative) {
    // Fully private -> registers.
    {
        Tile t("", Layout({}), {axis(Role::Private, Level::Group)}, /*reads=*/true, /*writes=*/false);
        EXPECT_FALSE(t.cooperative());
        EXPECT_EQ(t.required_space(), Space::Register);
    }

    // Block-cooperative (private grid axis) -> shared.
    {
        Tile t("", Layout({}), {axis(Role::Private, Level::Device), axis(Role::Cooperative, Level::Group)}, true, false);
        EXPECT_TRUE(t.cooperative());
        EXPECT_EQ(t.required_space(), Space::Shared);
        EXPECT_EQ(t.cooperative_axes().size(), 1u);
        EXPECT_EQ(t.private_axes().size(), 1u);
    }

    // Grid-cooperative dominates -> global.
    {
        Tile
            t("",
              Layout({}),
              {axis(Role::Cooperative, Level::Device), axis(Role::Cooperative, Level::Group)},
              true,
              false);
        EXPECT_EQ(t.required_space(), Space::Global);
    }

    // Warp-cooperative only -> registers (shuffle).
    {
        Tile t("", Layout({}), {axis(Role::Cooperative, Level::Subgroup)}, true, false);
        EXPECT_EQ(t.required_space(), Space::Register);
    }
}
