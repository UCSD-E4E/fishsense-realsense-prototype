#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>

#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>

using namespace std;
using namespace cv;
 
// helper function
// the scale of depth pixel to meter
float get_depth_scale(rs2::device dev) {
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors()) {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

// align depth image to color image
void align_Depth2Color(Mat depth, Mat color, rs2::pipeline_profile profile, int& cnt){
    // data stream
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
 
    // intrinsic matrices
    const rs2_intrinsics intrinsicDepth = depth_stream.get_intrinsics();
    const rs2_intrinsics intrinsicColor = color_stream.get_intrinsics();
 
    // extrinsic matrix from depth to color
    const rs2_extrinsics extrinsicDepth2Color = depth_stream.get_extrinsics_to(color_stream);

    // get the scale of depth pixel to meter（D455 is default to 0.001, i.e. 1mm）
    float depth_scale = get_depth_scale(profile.get_device());
    
    // 2d pixel points
    float depth_2d[2], color_2d[2];
    // 3d space points
    float depth_3d[3], color_3d[3];
    
    // initialize the result matrix
    // the result matrix is ideal to be the transformed depth image that can align with color image
    // i.e., the cols and rows info are from color frame, but the depth info is mapped from depth info
    // align this with color image, we can get the corresponding depth info of a color pixel in Mat result
    vector<vector<float>> result (color.rows, vector<float> (color.cols, 0));
    // 2d point in color pixel frame that was mapped from 2d point in depth pixel frame
    int new_row = 0, new_col = 0;

    // iterate through the depth image
    for(int row = 0; row < depth.rows; row++){
        for(int col = 0;col < depth.cols; col++){
	    // current depth points in pixel frame (2d)
            depth_2d[0] = col;
            depth_2d[1] = row;
            // get the depth of current depth point
            uint16_t depth_value = depth.at<uint16_t>(row, col);
            // change depth_value from pixel to meters
            float depth_m = depth_value * depth_scale;
	    // transform from depth 2D pixel frame to depth 3D camera frame
            rs2_deproject_pixel_to_point(depth_3d, &intrinsicDepth, depth_2d, depth_m);
            // transform from depth 3D camera frame to color 3D camera frame
            rs2_transform_point_to_point(color_3d, &extrinsicDepth2Color, depth_3d);
            // transform from color 3D camera frame to color 2D pixel frame
            rs2_project_point_to_pixel(color_2d, &intrinsicColor, color_3d);
 
            // get coordinates in color pixel frame after doing such mapping (2d)
            new_col = (int)color_2d[0];
            new_row = (int)color_2d[1];

	    // if the mapped points are out of boundary, discard
	    if (new_col < 0 || new_col > color.cols-1)
	        continue;
	    if (new_row < 0 || new_row > color.rows-1)
	        continue;

	    // in Mat result, write depth info to the point that is sucessfully mapped
	    result[new_row][new_col] = depth_m;
        }
    }

    // save color image and depth image (in different format)
    //imwrite(to_string(cnt)+".csv", result); // cause error here
    imwrite(to_string(cnt)+".png", color);

    ofstream depth_info(to_string(cnt)+".csv");
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[0].size(); j++) {
	    depth_info << result[i][j];
	    if (j < result[0].size()-1) {
	        depth_info << ",";
	    }
        }
	depth_info << endl;
    }
    
    cnt++;
}
 
int main() {
    cout << "program starts" << endl;
    int cnt = 0;
    const char* depth_window = "depth_Image";
    namedWindow(depth_window, WINDOW_AUTOSIZE);
    const char* color_window = "color_Image";
    namedWindow(color_window, WINDOW_AUTOSIZE);
 
    // Helper to colorize depth images
    rs2::colorizer c;                         
 
    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
 
    //Calling pipeline's start() without any additional parameters will start the first device
    // with its default streams.
    //The start function returns the pipeline profile which the pipeline used to start the device
    rs2::pipeline_profile profile = pipe.start(cfg);
 
    // get the scale of depth pixel to meter（D455 is default to 0.001, i.e. 1mm）
    float depth_scale = get_depth_scale(profile.get_device());
 
    // data stream
    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
 
    // intrinsic matrices
    const rs2_intrinsics intrinsicDepth = depth_stream.get_intrinsics();
    const rs2_intrinsics intrinsicColor = color_stream.get_intrinsics();
 
    // extrinsic matrix from depth to color
    const rs2_extrinsics extrinsicDepth2Color = depth_stream.get_extrinsics_to(color_stream);

    // save color intrinsic matrix for further calculation
    ofstream intrinsic_matrix("intrinsic_matrix.txt");    
    intrinsic_matrix << intrinsicColor.width << endl;
    intrinsic_matrix << intrinsicColor.height << endl;
    intrinsic_matrix << intrinsicColor.ppx << endl;
    intrinsic_matrix << intrinsicColor.ppy << endl;
    intrinsic_matrix << intrinsicColor.fx << endl;
    intrinsic_matrix << intrinsicColor.fy << endl;
    intrinsic_matrix << intrinsicColor.model << endl;
    for (int i = 0; i < 5; i++) {
        intrinsic_matrix << intrinsicColor.coeffs[i] << endl;;
    }
    
    // Application still alive?
    while (cvGetWindowHandle(depth_window) && cvGetWindowHandle(color_window)) {
        // we block the application until a frameset is available
        rs2::frameset frameset = pipe.wait_for_frames();
        // get the color frame and depth frame
        rs2::frame color_frame = frameset.get_color_frame();
        rs2::frame depth_frame = frameset.get_depth_frame();
	// this only for show (show depth frame in colorized form)
        rs2::frame depth_frame_c = frameset.get_depth_frame().apply_filter(c);
        // get the width and height for depth frame and color frame
        const int depth_w = depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = depth_frame.as<rs2::video_frame>().get_height();
        const int color_w = color_frame.as<rs2::video_frame>().get_width();
        const int color_h = color_frame.as<rs2::video_frame>().get_height();
 
        // change the frame into opencv Mat format
        Mat depth_image(Size(depth_w, depth_h),
                                CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        Mat depth_image_c(Size(depth_w, depth_h),
                                CV_8UC3, (void*)depth_frame_c.get_data(), Mat::AUTO_STEP);
        Mat color_image(Size(color_w, color_h),
                                CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        // align the depth image to the color image
        align_Depth2Color(depth_image, color_image, profile, cnt);
	
        // show
	// show the colorized depth image
        imshow(depth_window, depth_image_c);
        imshow(color_window, color_image);
        waitKey(10);
    }
    
    return 0;
}
