#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams
using namespace std;
#include <limits>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <thread>
#include <mutex>
#include <memory>
#include <functional>
#include <thread>
#include <string.h>
#include <chrono>
// 3rd party header for writing png files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <wiringPi.h>

#define PWR_CTRL 12
#define PWR_REED 13
#define REC_REED 14
#define PWR_LED 6
#define REC_LED 10
#define DEBUG_LED1 11
#define DEBUG_LED2 31

// Helper function for writing metadata to disk as a csv file
void metadata_to_csv(const rs2::frame& frm, const std::string& filename);

// This waits for the first 30 frames for exposure and continuously write the frames to disk.
// It can be useful for debugging an embedded system with no display.
int main(int argc, char* argv[]) try
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    wiringPiSetup();
    pinMode(REC_REED,INPUT); //REC_REED
    pullUpDnControl(REC_REED, PUD_UP);
    pinMode(PWR_CTRL,OUTPUT); //PWR_CTRL
    digitalWrite(PWR_CTRL,1);
    pinMode(PWR_REED,INPUT); //PWR_REED
    pullUpDnControl(PWR_REED, PUD_UP);
    pinMode(PWR_LED,OUTPUT); //PWR_LED
    digitalWrite(PWR_LED,1);
    pinMode(REC_LED,OUTPUT); //REC_LED
    digitalWrite(REC_LED,0);
    pinMode(DEBUG_LED1,OUTPUT); //DEBUG_LED
    digitalWrite(DEBUG_LED1,1);
    pinMode(DEBUG_LED2,OUTPUT);
    digitalWrite(DEBUG_LED2,0);
    
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;
    
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);
    
    int run_index = 0;
    fstream runfile;
    runfile.open("/home/pi/run.txt");
    runfile>>run_index;
    runfile.close();
    run_index++;
    std::stringstream dirprefix;
    dirprefix << "/home/pi/run" << run_index << "/";
    std::ofstream ofs;
    ofs.open("/home/pi/run.txt", std::ofstream::out | std::ofstream::trunc);
    ofs << run_index;
    ofs.close();
    int check;
    const std::string tmp = dirprefix.str();
    check = mkdir(tmp.c_str(),0777);
    if(check) {
        exit(1);
    }
    int frame_index = 0;
    bool rec_flag = false;
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
    while(digitalRead(PWR_REED)==0) {
        if(digitalRead(REC_REED)==0) {
            digitalWrite(REC_LED,1);
            // Declare depth colorizer for pretty visualization of depth data
            //rs2::colorizer color_map;
            
            // Declare RealSense pipeline, encapsulating the actual device and sensors
            rs2::pipeline pipe;
            // Start streaming with default recommended configuration
            if(!rec_flag) {
                digitalWrite(REC_LED,1);
                rec_flag=true;
                pipe.start(cfg);
                // Capture 30 frames to give autoexposure, etc. a chance to settle
                for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();
            }
            
            bool colorFlag = false;
            bool depthFlag = false;
            rs2::frame depthvf, colorvf;
            
            // Wait for the next set of frames from the camera. Now that autoexposure, etc.
            // has settled, we will write these to disk
            while(rec_flag) {
                for (auto&& frame : pipe.wait_for_frames())
                {
                    // We can only save video frames as pngs, so we skip the rest
                    if (auto vf = frame.as<rs2::video_frame>())
                    {
                        // Use the colorizer to get an rgb image for the depth stream
                        if (vf.is<rs2::depth_frame>()) {
                            depthFlag = true;
                            depthvf = vf;
                            //std::cout << "got depth\n";
                        } else {
                            colorFlag = true;
                            colorvf = vf;
                            //std::cout << "got color\n";
                        }
                        
                        // Write images to disk
                        if(depthFlag && colorFlag) {
                            rs2::video_frame cvf(colorvf);
                            rs2::depth_frame dvf(depthvf);
                            std::stringstream png_file;
                            png_file << dirprefix.str() << "img-" << cvf.get_profile().stream_name() << frame_index << ".png";
                            stbi_write_png(png_file.str().c_str(), cvf.get_width(), cvf.get_height(),
                                           cvf.get_bytes_per_pixel(), cvf.get_data(), cvf.get_stride_in_bytes());
                            std::cout << "Saved " << png_file.str() << std::endl;
                            
                            std::stringstream depthname;
                            depthname << dirprefix.str() << "img-" << dvf.get_profile().stream_name() << frame_index << ".csv";
                            std::ofstream fs(depthname.str(),std::ios::trunc);
                            if(fs) {
                                for(int y=0;y<dvf.get_height();y++) {
                                    auto delim = "";
                                    
                                    for(int x=0;x<dvf.get_width();x++) {
                                        fs << delim << dvf.get_distance(x,y);
                                        delim = ",";
                                    }
                                    fs << '\n';
                                }
                                fs.flush();
                            }
                            
                            depthFlag = false;
                            colorFlag = false;
                            frame_index++;
                        }
                        
                        // Record per-frame metadata for UVC streams
                        /*std::stringstream csv_file;
                         csv_file << "output-" << vf.get_profile().stream_name() << frame_index
                         << "-metadata.csv";
                         metadata_to_csv(vf, csv_file.str());*/
                    }
                }
                if(digitalRead(REC_REED)==0) {
                    digitalWrite(REC_LED,0);
                    rec_flag=false;
                    std::cout << "\nLED Down\n";
                    pipe.stop();
                    std::cout << "Pipe stopped\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
                
                //Prompt the user for next shot
                //cout << "Press Enter to Continue";
                //cin.ignore(std::numeric_limits<streamsize>::max(), '\n');
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    digitalWrite(DEBUG_LED1,0);
    system("sudo poweroff");
    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

void metadata_to_csv(const rs2::frame& frm, const std::string& filename)
{
    std::ofstream csv;
    
    csv.open(filename);
    
    //    std::cout << "Writing metadata to " << filename << endl;
    csv << "Stream," << rs2_stream_to_string(frm.get_profile().stream_type()) << "\nMetadata Attribute,Value\n";
    
    // Record all the available metadata attributes
    for (size_t i = 0; i < RS2_FRAME_METADATA_COUNT; i++)
    {
        if (frm.supports_frame_metadata((rs2_frame_metadata_value)i))
        {
            csv << rs2_frame_metadata_to_string((rs2_frame_metadata_value)i) << ","
            << frm.get_frame_metadata((rs2_frame_metadata_value)i) << "\n";
        }
    }
    
    csv.close();
}
