#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/out_local_storage.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/tile_fusion.h>

namespace sdfg {
namespace transformations {

Recorder::Recorder() : history_(nlohmann::json::array()) {}


void Recorder::save(std::filesystem::path path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving transformations: " + path.string());
    }
    file << history_.dump(4); // Pretty print with an indent of 4 spaces
    file.close();
}

} // namespace transformations
} // namespace sdfg
