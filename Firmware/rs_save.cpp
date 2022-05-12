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
#include <stdexcept>

//Saving firmware for FishSense prototype 2.0 (TX2)
//February 12th, 2022
//Peter Tueller ptueller@eng.ucsd.edu

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd,"r");
    if(!pipe) throw std::runtime_error("popen() failed!");
    try {
        while(fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch(...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

int main(int argc, char* argv[]) try
{
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH,1280,720,RS2_FORMAT_Z16,15);
    cfg.enable_stream(RS2_STREAM_COLOR,1280,720,RS2_FORMAT_RGB8,15);

    int run_index = 0;
    fstream runfile;
    runfile.open("/usr/local/share/FishSense/run.txt");
    runfile>>run_index;
    runfile.close();
    run_index++;
    std::stringstream dirprefix;
    dirprefix << "/mnt/data/" << run_index;
    std::ofstream ofs;
    ofs.open("/usr/local/share/FishSense/run.txt", std::ofstream::out | std::ofstream::trunc);
    ofs << run_index;
    ofs.close();
    int check;
    const std::string tmp = dirprefix.str();
    check = mkdir(tmp.c_str(),0777);
    if(check) {
        exit(1);
    }

    int bag_index = 0;
    bool rec_flag = false;

    std::stringstream bag_location;
    bag_location << dirprefix.str().c_str() << bag_index << ".bag";
    cfg.enable_record_to_file(bag_location.str().c_str());

    //Set up GPIOs
    system("echo 388 > /sys/class/gpio/export");
    cout << "388 export\n";
    system("echo 298 > /sys/class/gpio/export");
    cout << "298 export\n";
    system("echo in > /sys/class/gpio/gpio388/direction");
    cout << "388 in\n";
    system("echo out > /sys/class/gpio/gpio298/direction");
    cout << "298 out\n";
    system("echo 1 > /sys/class/gpio/gpio298/active_low");
    cout << "298 active low\n";

    //Blink the REC LED to prove that the system has booted and is running the program
    for(int i=0; i<5; i++) {
        system("echo 0 > /sys/class/gpio/gpio298/value");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        system("echo 1 > /sys/class/gpio/gpio298/value");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        cout << "blink\n";
    }

    while(true) {
        if(exec("cat /sys/class/gpio/gpio388/value") == "0\n") { //If we see a magnetic signal
            cout << "saw reed switch\n";
            rs2::pipeline pipe;
            if(!rec_flag) { //And are not already recording
                system("echo 0 > /sys/class/gpio/gpio298/value"); //Turn on REC LED
                rec_flag=true;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                pipe.start(cfg); //bag file should be saving at this point!
		cout << "saved bag to " << bag_location.str() << endl;
            }

            while(rec_flag) {
                //We are having fun and saving the bag, all is well in the world
                //Fun is continuing to be had
                if(exec("cat /sys/class/gpio/gpio388/value") == "0\n") { //If we see a magnetic signal again
                    system("echo 1 > /sys/class/gpio/gpio298/value"); //Turn off REC LED
                    rec_flag = false;
                    pipe.stop(); //This should save the bag file
                    //Prep the potential next bag file
                    bag_index++;
                    std::stringstream bag_location_new;
                    bag_location_new << dirprefix.str().c_str() << bag_index << ".bag";
                    cfg.enable_record_to_file(bag_location_new.str().c_str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000)); //Don't want to catch multiple magnetic switches
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    //We should never reach here
    system("reboot");
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
