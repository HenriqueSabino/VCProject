#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct Color
{
    uint8_t B, G, R;
};

bool isSkinPixel(Color p)
{
    double r = p.R / 255.0;
    double g = p.G / 255.0;

    double f1 = -1.376 * (r * r) + 1.0743 * r + 0.2;
    double f2 = -0.776 * (r * r) + 0.5601 * r + 0.18;

    double w = (r - 0.33) * (r - 0.33) + (g - 0.33) * (g - 0.33);

    double a = p.R - p.G;
    double b = p.R - p.B;
    double c = p.G - p.B;

    double theta = acos(0.5 * (a + b) / sqrt(a * a + b * c));

    double H = (p.B <= p.G) ? theta : 360 - theta;

    return (g > f2 && g < f1 && w > 0.001 && (H <= 20 || H > 240));
}

bool isHairPixel(Color p)
{
    double I = 1.0 / 3 * (p.R + p.G + p.B);

    double a = p.R - p.G;
    double b = p.R - p.B;
    double c = p.G - p.B;

    double theta = acos(0.5 * (a + b) / sqrt(a * a + b * c));

    double H = (p.B <= p.G) ? theta : 360 - theta;

    return (I < 80 && (p.B - p.G < 15 || p.B - p.R < 15)) || (H > 20 && H <= 40);
}

void detectSkinAndHair(Mat src, Mat skin, Mat hair, int groupWidth)
{
    // Going trough every fifth pixel
    for (int i = 0; i < src.rows - groupWidth; i += groupWidth)
    {
        if (i >= src.rows)
            break;

        for (int j = 0; j < src.cols - groupWidth; j += groupWidth)
        {
            if (j >= src.cols)
                break;

            int skinPixelCount = 0;
            int hairPixelCount = 0;

            // Running trough a groupWidth x groupWidth pixel grid to determine if the pixel is a skin or hair pixel
            for (int a = i; a < i + groupWidth; a++)
            {
                // Skipping edge cases
                if (a >= src.rows)
                    continue;

                Color *colors = src.ptr<Color>(a);

                for (int b = j; b < j + groupWidth; b++)
                {
                    // Skipping edge cases
                    if (b >= src.cols)
                        continue;

                    Color color = colors[b];

                    if (isSkinPixel(color))
                        skinPixelCount++;

                    if (isHairPixel(color))
                        hairPixelCount++;
                }
            }

            if (skinPixelCount >= groupWidth * groupWidth / 2)
                skin.ptr(i / groupWidth)[j / groupWidth] = 1;
            else

                skin.ptr(i / groupWidth)[j / groupWidth] = 0;

            if (hairPixelCount >= groupWidth * groupWidth / 2)
                hair.ptr(i / groupWidth)[j / groupWidth] = 1;
            else

                hair.ptr(i / groupWidth)[j / groupWidth] = 0;
        }
    }
}

void detectFaces(Mat src, vector<Rect> *faces, int groupWidth, int minArea)
{
    Mat skin = Mat(src.rows / groupWidth, src.cols / groupWidth, CV_8U);
    Mat hair = Mat(src.rows / groupWidth, src.cols / groupWidth, CV_8U);

    detectSkinAndHair(src, skin, hair, groupWidth);

    faces->clear();

    vector<Rect> skinBBox;
    vector<vector<Point>> skinContours;
    findContours(skin, skinContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < skinContours.size(); i++)
    {
        if (contourArea(skinContours[i]) >= minArea)
            skinBBox.push_back(boundingRect(skinContours[i]));
    }

    vector<Rect> hairBBox;
    vector<vector<Point>> hairContours;
    findContours(hair, hairContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < hairContours.size(); i++)
    {
        if (contourArea(hairContours[i]) >= minArea)
            hairBBox.push_back(boundingRect(hairContours[i]));
    }

    // Cheking how the boxes overlap
    for (int i = 0; i < skinBBox.size(); i++)
    {
        for (int j = 0; j < hairBBox.size(); j++)
        {
            // Matches cases: 0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 15
            if (skinBBox[i].x >= hairBBox[j].x)
            {
                // Matches cases: 0, 2, 3, 5, 6, 7, 11, 12, 15
                if (skinBBox[i].y >= hairBBox[j].y)
                {
                    // Matches cases: 1
                    if (skinBBox[i].y + skinBBox[i].height >= hairBBox[j].y + hairBBox[j].height)
                    {
                        faces->push_back(skinBBox[i]);
                        hairBBox.erase(hairBBox.begin() + j);
                        break;
                    }
                    else if (skinBBox[i].x + skinBBox[i].width <= hairBBox[j].x + hairBBox[j].width)
                    {
                        faces->push_back(skinBBox[i]);
                        hairBBox.erase(hairBBox.begin() + j);
                        break;
                    }
                }
                else if (skinBBox[i].x + skinBBox[i].width > hairBBox[j].x + hairBBox[j].width &&
                         skinBBox[i].y + skinBBox[i].height > hairBBox[j].y + hairBBox[j].height) // Matches cases: 9
                {
                    faces->push_back(skinBBox[i]);
                    hairBBox.erase(hairBBox.begin() + j);
                    break;
                }
            }
            else // Other cases, except 8
            {
                if (skinBBox[i].y >= hairBBox[j].y)
                {
                    if (skinBBox[i].y + skinBBox[i].height > hairBBox[j].y)
                    {
                        faces->push_back(skinBBox[i]);
                        hairBBox.erase(hairBBox.begin() + j);
                        break;
                    }
                }
            }
        }
    }

    // Resizing rects
    for (int i = 0; i < faces->size(); i++)
    {
        faces->at(i).x *= groupWidth;
        faces->at(i).y *= groupWidth;
        faces->at(i).width *= groupWidth;
        faces->at(i).height *= groupWidth;

        if (faces->at(i).x + faces->at(i).width > src.cols)
        {
            faces->at(i).width = src.cols - faces->at(i).x;
        }

        if (faces->at(i).y + faces->at(i).height > src.rows)
        {
            faces->at(i).height = src.rows - faces->at(i).y;
        }
    }
}

int main()
{
    VideoCapture cap = VideoCapture(0);

    while (true)
    {
        Mat src;
        cap >> src;

        int groupWidth = 7, minArea = 20;

        vector<Rect> faces;
        detectFaces(src, &faces, groupWidth, minArea);

        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(src, faces.at(i), Scalar(0, 0, 255));
        }

        imshow("detection", src);

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}