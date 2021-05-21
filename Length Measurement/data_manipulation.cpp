#include <iostream>
#include <math.h>
#include <string>
#include <librealsense2/rsutil.h>

/*

test by ./0512 test2/9.csv and 9.png
head and tail clusters:
230 241 0.395
229 240 0.395
230 245 0.397
537 219 0.381
533 218 0.382
530 215 0.381
[(230, 241), (229, 240), (230, 245)]
[(537, 219), (533, 218), (530, 215)]

average value for head and tail:
head: (230, 242, 0.396)
tail: (533, 217, 0.381)

*/

//int width, int height, float ppx, float ppy, float fx, float fy, rs2_distortion model, float coeffs[5]


void deproject_pixel_to_point(float point[3], int width, int height, float ppx, float ppy, float fx, float fy, std::string model, float coeffs[5], float pixel[2], float depth)
{
    assert(model != "RS2_DISTORTION_MODIFIED_BROWN_CONRADY"); // Cannot deproject from a forward-distorted image
    //assert(model != "RS2_DISTORTION_BROWN_CONRADY"); // Cannot deproject to an brown conrady model

    float x = (pixel[0] - ppx) / fx;
    float y = (pixel[1] - ppy) / fy;

    float xo = x;
    float yo = y;

    if(model == "RS2_DISTORTION_INVERSE_BROWN_CONRADY")
    {
        // need to loop until convergence 
        // 10 iterations determined empirically
        for (int i = 0; i < 10; i++)
        {
            float r2 = x * x + y * y;
            float icdist = (float)1 / (float)(1 + ((coeffs[4] * r2 + coeffs[1])*r2 + coeffs[0])*r2);
            float xq = x / icdist;
            float yq = y / icdist;
            float delta_x = 2 * coeffs[2] * xq*yq + coeffs[3] * (r2 + 2 * xq*xq);
            float delta_y = 2 * coeffs[3] * xq*yq + coeffs[2] * (r2 + 2 * yq*yq);
            x = (xo - delta_x)*icdist;
            y = (yo - delta_y)*icdist;
        }
    }
    if (model == "RS2_DISTORTION_BROWN_CONRADY")
    {
        // need to loop until convergence 
        // 10 iterations determined empirically
        for (int i = 0; i < 10; i++)
        {
            float r2 = x * x + y * y;
            float icdist = (float)1 / (float)(1 + ((coeffs[4] * r2 + coeffs[1])*r2 + coeffs[0])*r2);
            float delta_x = 2 * coeffs[2] * x*y + coeffs[3] * (r2 + 2 * x*x);
            float delta_y = 2 * coeffs[3] * x*y + coeffs[2] * (r2 + 2 * y*y);
            x = (xo - delta_x)*icdist;
            y = (yo - delta_y)*icdist;
        }
        
    }
    if (model == "RS2_DISTORTION_KANNALA_BRANDT4")
    {
        float rd = sqrtf(x*x + y*y);
        if (rd < FLT_EPSILON)
        {
            rd = FLT_EPSILON;
        }

        float theta = rd;
        float theta2 = rd*rd;
        for (int i = 0; i < 4; i++)
        {
            float f = theta*(1 + theta2*(coeffs[0] + theta2*(coeffs[1] + theta2*(coeffs[2] + theta2*coeffs[3])))) - rd;
            if (fabs(f) < FLT_EPSILON)
            {
                break;
            }
            float df = 1 + theta2*(3 * coeffs[0] + theta2*(5 * coeffs[1] + theta2*(7 * coeffs[2] + 9 * theta2*coeffs[3])));
            theta -= f / df;
            theta2 = theta*theta;
        }
        float r = tan(theta);
        x *= r / rd;
        y *= r / rd;
    }
    if (model == "RS2_DISTORTION_FTHETA")
    {
        float rd = sqrtf(x*x + y*y);
        if (rd < FLT_EPSILON)
        {
            rd = FLT_EPSILON;
        }
        float r = (float)(tan(coeffs[0] * rd) / atan(2 * tan(coeffs[0] / 2.0f)));
        x *= r / rd;
        y *= r / rd;
    }

    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}


int main() {
    int width = 640;
    int height = 480;
    float ppx = 310.115;
    float ppy = 242.869;
    float fx = 382.465;
    float fy = 382.084;
    std::string model = "Inverse Brown Conrady";
    float coeffs[5] = {-0.056538, 0.0664504, 0.000127996, -0.000611001, -0.0212887};

    float point1[3];
    float point2[3];
    float pixel1[2] = {230, 242};
    float pixel2[2] = {533, 217};
    float depth1 = 0.396;
    float depth2 = 0.381;
    
    deproject_pixel_to_point(point1, width, height, ppx, ppy, fx, fy, model, coeffs, pixel1, depth1);
    deproject_pixel_to_point(point2, width, height, ppx, ppy, fx, fy, model, coeffs, pixel2, depth2);
    
    std::cout << point1[0] << " " << point1[1] << " " << point1[2] << std::endl;
    std::cout << point2[0] << " " << point2[1] << " " << point2[2] << std::endl;
    float dis = sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2) + pow(point1[2]-point2[2], 2));
    std::cout << dis << std::endl;

      
    return 0;
 }


// read in csv file
// save as Mat/vector of vector
// points needed are read from color image

// read in intrinsic matrix line by line in string 
// and then change it back to the original type
// i.e., int, float

// transform from color 2D pixel frame to color 3D camera frame
// remember depth here is from the Mat result (mapped depth image)
//            rs2_deproject_pixel_to_point

// transform from depth 3D camera frame to world 3D frame
// as the origin of color camera frame is the same as that of world frame
// Rotation is just [1,0,0; 0,1,0;0,0,1], Translation is [0;0;0]
// so the 3d coordinates in color frame is just the world coordinates
