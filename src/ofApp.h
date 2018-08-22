#pragma once
#include "ofMain.h"
#include "ofxCsv.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
    struct voxel{
        int x;
        int y;
        int z;
        int col;
    };
    struct domain{
        int x;
        int y;
        int z;
        int length(){
            return sqrt(x*x+y*y+z*z);
        }
    };
    
    ofEasyCam cam;
    ofxCsv csv;
    
    uint64 voxelSize;
    uint64 colorSize;
    domain dist;
    
    vector<voxel> voxels;
    vector<ofColor> colorMap;
    vector<vector<voxel>> voxelLayer;
    
    vector<vector<vector<int>>> hull;
    vector<vector<vector<int>>> core;
    
    bool exporting;
    int currentLayer;
    ofPixels pix;
    ofPixels corePix;
    ofPixels contourPix;
    cv::Mat distMap;
    ofImage resImg;
    ofxCvGrayscaleImage grayImg;
    ofxCvGrayscaleImage contourImg;
    ofxCv::ContourFinder contourFinder;
    ofColor coreCol;
    ofColor newCol;
    int hullDepth;
    int fillRad;
    string name;
    float elTime;
    int LUT[255][255][255];
    const int cmykw[5][3] = {{0,255,255},{255,0,255},{255,255,0},{0,0,0},{255,255,255}};
//    vector<ofColor> cmykw;
    float weight[4];
};
