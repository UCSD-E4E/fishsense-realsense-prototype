#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>
using namespace std;
#include <limits>

int main(int argc, char * argv[]) try
{
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    std::cout << "Attempting to open pipe...\n";
    pipe.start();
    std::cout << "Pipe started\n";

    rs2::colorizer colorizer;

    for(int frame_index = 0; true; frame_index++)
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();
        auto depth = frames.get_depth_frame();


        auto colorized = colorizer.colorize(depth);

        pc.map_to(colorized);
        points = pc.calculate(depth);

        std::ostringstream frame_name;
        frame_name << "frame_" << frame_index << ".ply";

        std::cout << "Saving: " << frame_name.str() << "\n";

        points.export_to_ply(frame_name.str(), colorized);

        //Prompt the user for next shot
        cout << "Press Enter to Continue";
        cin.ignore(std::numeric_limits<streamsize>::max(), '\n');
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}