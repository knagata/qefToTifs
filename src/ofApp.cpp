#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void ofApp::setup(){
    cout << "run" << endl;
    ///////////////////////////////////
    //  load csv to make voxel list  //
    ///////////////////////////////////
    if(csv.load("dog.qef", " ")) {
        colorSize = csv[4].getInt(0);
        cout<<"load header"<<endl;
        for (uint64 i=5; i<colorSize+5; i++) {
            colorMap.push_back(ofColor(255*csv[i].getFloat(0),255*csv[i].getFloat(1), 255*csv[i].getFloat(2)));
        }
        cout<<"load color"<<endl;
        for (uint64 i=colorSize+5; i<csv.size()-1; i++) {
            voxels.push_back(voxel{csv[i].getInt(0), csv[i].getInt(1), csv[i].getInt(2), csv[i].getInt(3)});
        }
        cout<<"load voxel"<<endl;
        
        int xmax = 0;
        int ymax = 0;
        int zmax = 0;
        int height = 0;
        xmax = voxels[voxels.size()-1].x;
        vector<voxel> layer;
        for (uint64 i=0; i<voxels.size(); i++) {
            if (zmax < voxels[i].z){ zmax=voxels[i].z; }
            //if (ymax < voxels[i].y){ ymax=voxels[i].y; }
            //if (xmax < voxels[i].x) xmax=voxels[i].x;
            if (voxels[i].x>height) {
                if (ymax < voxels[i-1].y) ymax=voxels[i-1].y;
                cout << "reading layer "<<ofToString(height)<<" of "<<ofToString(xmax)<<endl;
                voxelLayer.push_back(layer);
                layer.clear();
                height++;
            }
            layer.push_back(voxels[i]);
        }
        voxelLayer.push_back(layer);
        layer.clear();
        dist = domain{xmax, ymax, zmax};
        cout<<"generate layer"<<endl;
        
        cout << "boundings: " << dist.x<<","<<dist.y<<","<<dist.z << endl;
        cout << "voxel: " << voxels.size() << endl;
        cout << "layer: " << voxelLayer.size() << endl;
        cout << "color: " << colorSize << endl;
    }else{
        cout<<"failed to load"<<endl;
    }
    
    ////////////////
    //generate LUT//
    ////////////////
//    cmykw.push_back(ofColor(0,255,255));
//    cmykw.push_back(ofColor(255,0,255));
//    cmykw.push_back(ofColor(255,255,0));
//    cmykw.push_back(ofColor(0,0,0));
//    cmykw.push_back(ofColor(255,255,255));
//    int cmykwr[5] = {0,255,255,0,255};
//    int cmykwg[5] = {255,0,255,0,255};
//    int cmykwb[5] = {255,255,0,0,255};
    weight[0] = 7/16;
    weight[1] = 3/16;
    weight[2] = 5/16;
    weight[3] = 1/16;
//    vector<float> dcmykw = vector<float>(5);
    for (int i=0; i<256; i++) {
        for (int j=0; j<256; j++) {
            for (int k=0; k<256; k++) {
                long closestDist=pow(255,2)*3;
                int closestIndex=0;
                long dist;
                for (int m=0; m<5; m++) {
                    dist = pow(i-cmykw[m][0],2)+pow(j-cmykw[m][1],2)+pow(k-cmykw[m][2],2);
                    if (dist<closestDist) {
                        closestIndex=m;
                        closestDist=dist;
                    }
//                    dcmykw[m]= ofVec3f(cmykw[m].r,cmykw[m].g,cmykw[m].b).distance(ofVec3f(i, j, k));
                }
//                std::vector<float>::iterator minIt = std::min_element(dcmykw.begin(), dcmykw.end());
//                LUT[i][j][k] = std::distance(dcmykw.begin(), minIt);
                LUT[i][j][k] = closestIndex;
            }
        }
    }
    cout<<"generate LUT"<<endl;
    
    cam.setDistance(dist.length());
    coreCol = ofColor(0,255,255);
    newCol = ofColor(255,255,255,255);
    hullDepth = 20;
    fillRad = 2;
    exporting = false;
//    ofSetFrameRate(5);
}

//--------------------------------------------------------------
void ofApp::update(){
    if (exporting) {
        int w = dist.y;
        int h = dist.z;
        
        pix.clear();
        pix.allocate(w, h, OF_PIXELS_RGBA);
        corePix.clear();
        corePix.allocate(w, h, OF_PIXELS_MONO);
        contourPix.clear();
        contourPix.allocate(w, h, OF_PIXELS_GRAY_ALPHA);
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                pix.setColor(i, j, ofColor(0,0,0,0));
                corePix.setColor(i,j, ofColor(0));
                contourPix.setColor(i, j, ofColor(0,0));
            }
        }
        
        for (int i=0; i<voxelLayer[currentLayer].size(); i++) {
            int y = voxelLayer[currentLayer][i].y;
            int z = voxelLayer[currentLayer][i].z;
            ofColor col = colorMap[voxelLayer[currentLayer][i].col];
            if(col.r==coreCol.r&&col.g==coreCol.g&&col.b==coreCol.b){
                corePix.setColor(y, z, ofColor(255));
            }else{
                pix.setColor(y, z, ofColor(col.r, col.g, col.b, 255));
            }
        }
        
        /////////////////
        //find contours//
        /////////////////
        grayImg.clear();
        grayImg.allocate(w, h);
        grayImg.setFromPixels(corePix);
        contourFinder.setSimplify(false);
        contourFinder.setMinAreaRadius(0);
        contourFinder.setMaxAreaRadius(500);
        contourFinder.setThreshold(1);
        contourFinder.findContours(grayImg);
        contourFinder.setFindHoles(false);
        
        /////////////////////
        //find contour dist//
        /////////////////////
        contourImg.clear();
        contourImg.allocate(w, h);
        distMap = Mat(w, h, CV_32FC1, Scalar_<float>(-1));
        for (int c=0; c<contourFinder.getContours().size(); c++) {
            vector<cv::Point> contour = contourFinder.getContour(c);
            for (int i=0; i<w; i++) {
                for (int j=0; j<h; j++) {
                    float d = pointPolygonTest(contour, cv::Point2f(i,j), true);
                    if (d>-1) {
                        distMap.at<float>(i,j) = d;
                    }
                }
            }
        }
        double minVal; double maxVal;
        minMaxLoc( distMap, &minVal, &maxVal, 0, 0, Mat() );
        minVal = abs(minVal); maxVal = abs(maxVal);
        int d=0;
        while (d<=hullDepth&&d<=maxVal) {
            for (int i=0; i<w; i++) {
                for (int j=0; j<h; j++) {
                    if (distMap.at<float>(i,j)>=d && distMap.at<float>(i,j)<d+1) {
                        int num=0;
                        int r=0;
                        int g=0;
                        int b=0;
                        for (int disp=fillRad; disp>0; disp--) {
                            for (int rad=0; rad<disp; rad++) {
                                ofColor rc = pix.getColor(i-disp+rad, j-rad);
                                ofColor lc = pix.getColor(i+disp-rad, j+rad);
                                ofColor uc = pix.getColor(i-rad, j-disp+rad);
                                ofColor dc = pix.getColor(i+rad, j+disp-rad);
                                if (rc.a>0){ num++; r+=rc.r; g+=rc.g; b+=rc.b;}
                                if (lc.a>0){ num++; r+=lc.r; g+=lc.g; b+=lc.b;}
                                if (uc.a>0){ num++; r+=uc.r; g+=uc.g; b+=uc.b;}
                                if (dc.a>0){ num++; r+=dc.r; g+=dc.g; b+=dc.b;}
                            }
                        }
//                        for (int dx=-fillRad; dx<=fillRad; dx++) {
//                            for (int dy=-fillRad; dy<=fillRad; dy++){
//                                ofColor tempc = pix.getColor(i+dx, j+dy);
//                                if (tempc.a>0){ num++; r+=tempc.r; g+=tempc.g; b+=tempc.b;}
//                            }
//                        }
                        if (num!=0) {
                            pix.setColor(i,j, ofColor(r/num, g/num, b/num, 255));
                        }
                    }
                }
            }
            d++;
        }
        
        /////////////
        //dithering//
        /////////////
//        for (int i=0; i<w-1; i++) {
//            for (int j=0; j<h-1; j++) {
//                ofColor oldCol = pix.getColor(i, j);
//                if (oldCol.a>0) {
//                    int newPixr,newPixg,newPixb;
//                    if (oldCol.r<128) newPixr=0; else newPixr=255;
//                    if (oldCol.g<128) newPixg=0; else newPixg=255;
//                    if (oldCol.b<128) newPixb=0; else newPixb=255;
//                    pix.setColor(i, j, ofColor(newPixr,newPixb,newPixg,oldCol.a));
//                    int quant_errorR = oldCol.r-newPixr;
//                    int quant_errorG = oldCol.g-newPixg;
//                    int quant_errorB = oldCol.b-newPixb;
//                    cout<<quant_errorR<<" "<<quant_errorG<<" "<<quant_errorB<<endl;
//                    pix.setColor(i+1,j, ofColor(pix.getColor(i+1,j).r + weight[0] * quant_errorR,
//                                                pix.getColor(i+1,j).g + weight[0] * quant_errorG,
//                                                pix.getColor(i+1,j).b + weight[0] * quant_errorB,
//                                                pix.getColor(i+1,j).a));
//                    pix.setColor(i-1,j+1, ofColor(pix.getColor(i-1,j+1).r + weight[1] * quant_errorR,
//                                                  pix.getColor(i-1,j+1).g + weight[1] * quant_errorG,
//                                                  pix.getColor(i-1,j+1).b + weight[1] * quant_errorB,
//                                                  pix.getColor(i-1,j+1).a));
//                    pix.setColor(i,j+1, ofColor(pix.getColor(i,j+1).r + weight[2] * quant_errorR,
//                                                pix.getColor(i,j+1).g + weight[2] * quant_errorG,
//                                                pix.getColor(i,j+1).b + weight[2] * quant_errorB,
//                                                pix.getColor(i,j+1).a));
//                    pix.setColor(i+1,j+1, ofColor(pix.getColor(i+1,j+1).r + weight[3] * quant_errorR,
//                                                  pix.getColor(i+1,j+1).g + weight[3] * quant_errorG,
//                                                  pix.getColor(i+1,j+1).b + weight[3] * quant_errorB,
//                                                  pix.getColor(i+1,j+1).a));
//                }
//            }
//        }
        
        
//        ofVec4f pixVec[w][h];
//        for (int i=0; i<h; i++) {
//            for (int j=0; j<w; j++) {
//                pixVec[i][j].x = pix.getColor(i, j).r;
//                pixVec[i][j].y = pix.getColor(i, j).g;
//                pixVec[i][j].z = pix.getColor(i, j).b;
//                pixVec[i][j].w = pix.getColor(i, j).a;
//            }
//        }
//        for (int i=0; i<w-1; i++) {
//            for (int j=0; j<h-1; j++) {
//                ofVec4f oldCol = pixVec[i][j];
//                if (oldCol.w>0) {
//                    Byte cx,cy,cz;
//                    cx=oldCol.x;
//                    cy=oldCol.y;
//                    cz=oldCol.z;
//                    ofVec4f newCol = cmykw[LUT[cx][cy][cz]];
//                    cout<<newCol.x<<endl;
//                    pixVec[i][j] = newCol;
//                    int quant_errorX = oldCol.x-newCol.x;
//                    int quant_errorY = oldCol.y-newCol.y;
//                    int quant_errorZ = oldCol.z-newCol.z;
//                    pixVec[i+1][j] = ofVec4f(pixVec[i+1][j].x + weight[0] * quant_errorX,
//                                             pixVec[i+1][j].y + weight[0] * quant_errorY,
//                                             pixVec[i+1][j].z + weight[0] * quant_errorZ,
//                                             pixVec[i+1][j].w);
//                    pixVec[i-1][j+1] = ofVec4f(pixVec[i-1][j+1].x + weight[1] * quant_errorX,
//                                               pixVec[i-1][j+1].y + weight[1] * quant_errorY,
//                                               pixVec[i-1][j+1].z + weight[1] * quant_errorZ,
//                                               pixVec[i-1][j+1].w);
//                    pixVec[i][j+1] = ofVec4f(pixVec[i][j+1].x + weight[2] * quant_errorX,
//                                             pixVec[i][j+1].y + weight[2] * quant_errorY,
//                                             pixVec[i][j+1].z + weight[2] * quant_errorZ,
//                                             pixVec[i][j+1].w);
//                    pixVec[i+1][j+1] = ofVec4f(pixVec[i+1][j+1].x + weight[3] * quant_errorX,
//                                               pixVec[i+1][j+1].y + weight[3] * quant_errorY,
//                                               pixVec[i+1][j+1].z + weight[3] * quant_errorZ,
//                                               pixVec[i+1][j+1].w);
//                }
//            }
//        }
//        for (int i=0; i<h; i++) {
//            for (int j=0; j<w; j++) {
//                pix.setColor(i, j, ofColor(Byte(pixVec[i][j].x),Byte(pixVec[i][j].y) ,Byte(pixVec[i][j].z), Byte(pix.getColor(i, j).a)));
//            }
//        }
        
        
        for (int i=0; i<w-1; i++) {
            for (int j=0; j<h-1; j++) {
                ofColor oldCol = pix.getColor(i, j);
                if (oldCol.a>0) {
                    int lut = LUT[oldCol.r][oldCol.g][oldCol.b];
//                    cout<<cmykw[lut][0]<<","<<cmykw[lut][1]<<","<<cmykw[lut][2]<<endl<<static_cast<unsigned>(oldCol.r)<<","<<static_cast<unsigned>(oldCol.g)<<","<<static_cast<unsigned>(oldCol.b)<<": "<<lut<<endl;
                    ofColor newCol;
                    switch (lut) {
                        case 0:
                            newCol = ofColor(0,255,255);
                            break;
                        case 1:
                            newCol = ofColor(255,0,255);
                            break;
                        case 2:
                            newCol = ofColor(255,255,0);
                            break;
                        case 3:
                            newCol = ofColor(0,0,0);;
                            break;
                        default:
                            newCol = ofColor(255,255,255);
                            break;
                    }
                    pix.setColor(i, j, newCol);
                    int quant_errorR = oldCol.r-newCol.r;
                    int quant_errorG = oldCol.g-newCol.g;
                    int quant_errorB = oldCol.b-newCol.b;
                    float newVals[3];

                    newVals[0] = pix.getColor(i+1,j).r + weight[0] * quant_errorR;
                    newVals[1] = pix.getColor(i+1,j).g + weight[0] * quant_errorG;
                    newVals[2] = pix.getColor(i+1,j).b + weight[0] * quant_errorB;
                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
                    pix.setColor(i+1,j, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i+1,j).a));
                    newVals[0] = pix.getColor(i-1,j+1).r + weight[1] * quant_errorR;
                    newVals[1] = pix.getColor(i-1,j+1).g + weight[1] * quant_errorG;
                    newVals[2] = pix.getColor(i-1,j+1).b + weight[1] * quant_errorB;
                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
                    pix.setColor(i-1,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i-1,j+1).a));
                    newVals[0] = pix.getColor(i,j+1).r + weight[2] * quant_errorR;
                    newVals[1] = pix.getColor(i,j+1).g + weight[2] * quant_errorG;
                    newVals[2] = pix.getColor(i,j+1).b + weight[2] * quant_errorB;
                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
                    pix.setColor(i,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i,j+1).a));
                    newVals[0] = pix.getColor(i+1,j+1).r + weight[3] * quant_errorR;
                    newVals[1] = pix.getColor(i+1,j+1).g + weight[3] * quant_errorG;
                    newVals[2] = pix.getColor(i+1,j+1).b + weight[3] * quant_errorB;
                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
                    pix.setColor(i+1,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i+1,j+1).a));
                }
            }
        }
        
        
        
//        for (int i=0; i<w; i++) {
//            for (int j=0; j<h; j++) {
//                if (distMap.at<float>(i,j)>-1) {
//                    contourPix.setColor(i, j, ofColor(distMap.at<float>(i,j)*255/maxVal, 255));
//                }
//            }
//        }
//        contourImg.setFromPixels(contourPix);
        
        resImg.clear();
        resImg.allocate(w, h, OF_IMAGE_COLOR_ALPHA);
        resImg.setFromPixels(pix);
        resImg.save(name+"_"+ofToString(currentLayer)+".tiff");
        if(currentLayer==voxelLayer.size()-1){
            exporting=false;
            cout << "time: " << ofGetElapsedTimef()-elTime << endl;
        }else{
            currentLayer++;
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    if (exporting) {
        ofBackground(255);
        ofSetColor(255);
        
        ofTexture tex;
        tex.loadData(pix);
        tex.draw(20, 20, dist.y*2, dist.z*2);
        
        ////////////////
        //draw contour//
        ////////////////
//        ofSetColor(255, 0, 0);
//        ofSetLineWidth(1);
//        ofPushMatrix();
//        ofTranslate(20, 20);
//        ofScale(2, 2);
//        contourFinder.draw();
//        ofPopMatrix();
        
        ofSetColor(0);
        ofDrawBitmapString("exporting... "+ofToString(currentLayer-1)+" of "+ofToString(voxelLayer.size())+" layers", 5, 15);
    }else{
        ofEnableDepthTest();
        cam.begin();
        ofPushMatrix();
        ofTranslate(-dist.x/2,-dist.y/2,-dist.z/2);
        for (int i=0; i<voxelLayer.size(); i++) {
            for (int j=0; j<voxelLayer[i].size(); j++) {
                ofSetColor(colorMap[(int)voxelLayer[i][j].col]);
                ofDrawBox(voxelLayer[i][j].x, voxelLayer[i][j].y, voxelLayer[i][j].z, 1);
            }
        }
        ofPopMatrix();
        cam.end();
        ofDisableDepthTest();
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == ' '){
        name = ofSystemTextBoxDialog("save?","result/penguin");
        exporting = true;
        elTime = ofGetElapsedTimef();
        currentLayer = 0;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
