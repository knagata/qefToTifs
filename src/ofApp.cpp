#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

//--------------------------------------------------------------
void ofApp::setup(){
    cout << "run" << endl;
    ///////////////////////////////////
    //  load csv to make voxel list  //
    ///////////////////////////////////
    if(csv.load("fiji.qef", " ")) {
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
        dist = domain{xmax+1, ymax+1, zmax+1};
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
//    for (int i=0; i<256; i++) {
//        for (int j=0; j<256; j++) {
//            for (int k=0; k<256; k++) {
//                long closestDist=pow(255,2)*3;
//                int closestIndex=0;
//                long dist;
//                for (int m=0; m<5; m++) {
//                    dist = pow(i-cmykw[m][0],2)+pow(j-cmykw[m][1],2)+pow(k-cmykw[m][2],2);
//                    if (dist<closestDist) {
//                        closestIndex=m;
//                        closestDist=dist;
//                    }
//                }
//                LUT[i][j][k] = closestIndex;
//            }
//        }
//    }
//    cout<<"generate LUT"<<endl;
    
    cam.setDistance(dist.length());
    coreCol = ofColor(0,255,255);
    newCol = ofColor(255,255,255,255);
    hullDepth = 10;
    for(int r=2;r<=hullDepth+2; r++){
        vector<ofVec2f> pts;
        for (float t=0; t<2.0; t+=0.25/r) {
            ofVec2f pt = ofVec2f(round(r*cos(t*PI)),round(r*sin(t*PI)));
            if (t==0) {
                pts.push_back(pt);
            }else if(pts[pts.size()-1].x!=pt.x&&pts[pts.size()-1].y!=pt.y && pts[0].x!=pt.x&&pts[0].y!=pt.y){
                pts.push_back(pt);
            }
        }
        cirPts.push_back(pts);
    }
//    fillRad = 2;
    exporting = false;
//    ofSetFrameRate(5);
}

//--------------------------------------------------------------
void ofApp::update(){
    if (exporting && currentLayer<voxelLayer.size()) {
        int w = dist.y;
        int h = dist.z;
        
        pix.clear();
        pix.allocate(w, h, OF_PIXELS_RGBA);
        corePix.clear();
        corePix.allocate(w, h, OF_PIXELS_MONO);
        newPix.clear();
        newPix.allocate(w, h, OF_PIXELS_RGBA);
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                pix.setColor(i, j, ofColor(0,0,0,0));
                corePix.setColor(i,j, ofColor(0));
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
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                newPix.setColor(i,j, ofColor(0,0,0,0));
            }
        }
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                int contDist = round(distMap.at<float>(i,j));
                if (contDist<=hullDepth&&contDist>=0) {
                    int num=0;
                    int r=0;
                    int g=0;
                    int b=0;
                    float minDist=(contDist+1)*2;
                    for (int x=-contDist-1; x<=contDist+1; x++) {
                        for (int y=-contDist-1; y<=contDist+1; y++) {
                            if (i+x>0&&i+x<w && j+y>0&&j+y<h) {
                                ofColor c = pix.getColor(i+x, j+y);
                                if (c.a>0) {
                                    float d = sqrt(x*x+y*y);
                                    if (d<minDist) {
                                        minDist=d;
                                        r=c.r;
                                        g=c.g;
                                        b=c.b;
                                    }
//                                    num++;
//                                    r+=c.r;
//                                    g+=c.g;
//                                    b+=c.b;
                                }
                            }
                        }
                    }
                    if (minDist<(contDist+1)*2) {
                        newPix.setColor(i, j, ofColor(r, g, b, 255));
                    }
//                    if (num>0) {
//                        newPix.setColor(i, j, ofColor(r/num, g/num, b/num, 255));
//                    }
//                    for (int offset=0; offset<=hullDepth-contDist; offset++) {
//                        for (int p=0; p<cirPts[contDist+offset].size(); p++) {
//                            if(cirPts[contDist][p].x+i>0&&cirPts[contDist][p].y+j>0 && cirPts[contDist][p].x+i<w&&cirPts[contDist][p].y+j<h){
//                                if (pix.getColor(cirPts[contDist][p].x+i, cirPts[contDist][p].y+j).a>0) {
//                                    num++;
//                                    r+=pix.getColor(cirPts[contDist][p].x+i, cirPts[contDist][p].y+j).r;
//                                    g+=pix.getColor(cirPts[contDist][p].x+i, cirPts[contDist][p].y+j).g;
//                                    b+=pix.getColor(cirPts[contDist][p].x+i, cirPts[contDist][p].y+j).b;
//                                }
//                            }
//                        }
//                        if (num>0) {
//                            cout<<num<<endl;
//                            newPix.setColor(i,j, ofColor(r/num, g/num, b/num, 255));
//                            break;
//                        }
//                    }
                }
            }
        }
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                if (newPix.getColor(i, j).a>0) {
                    pix.setColor(i,j, newPix.getColor(i, j));
                }
            }
        }
//        int d=0;
//        while (d<=hullDepth) {
//            for (int i=0; i<w; i++) {
//                for (int j=0; j<h; j++) {
//                    if (distMap.at<float>(i,j)>=d && distMap.at<float>(i,j)<d+1) {
//                        int num=0;
//                        int r=0;
//                        int g=0;
//                        int b=0;
//                        for (int disp=fillRad; disp>0; disp--) {
//                            for (int rad=0; rad<disp; rad++) {
//                                ofColor rc = pix.getColor(i-disp+rad, j-rad);
//                                ofColor lc = pix.getColor(i+disp-rad, j+rad);
//                                ofColor uc = pix.getColor(i-rad, j-disp+rad);
//                                ofColor dc = pix.getColor(i+rad, j+disp-rad);
//                                if (rc.a>0){ num++; r+=rc.r; g+=rc.g; b+=rc.b;}
//                                if (lc.a>0){ num++; r+=lc.r; g+=lc.g; b+=lc.b;}
//                                if (uc.a>0){ num++; r+=uc.r; g+=uc.g; b+=uc.b;}
//                                if (dc.a>0){ num++; r+=dc.r; g+=dc.g; b+=dc.b;}
//                            }
//                        }
//                        if (num!=0) {
//                            pix.setColor(i,j, ofColor(r/num, g/num, b/num, 255));
//                        }
//                    }
//                }
//            }
//            d++;
//        }
        
        /////////////
        //dithering//
        /////////////
//        for (int i=0; i<w-1; i++) {
//            for (int j=0; j<h-1; j++) {
//                ofColor oldCol = pix.getColor(i, j);
//                if (oldCol.a>0) {
//                    int lut = LUT[oldCol.r][oldCol.g][oldCol.b];
//                    //cout<<cmykw[lut][0]<<","<<cmykw[lut][1]<<","<<cmykw[lut][2]<<endl<<static_cast<unsigned>(oldCol.r)<<","<<static_cast<unsigned>(oldCol.g)<<","<<static_cast<unsigned>(oldCol.b)<<": "<<lut<<endl;
//                    ofColor newCol;
//                    switch (lut) {
//                        case 0:
//                            newCol = ofColor(0,255,255);
//                            break;
//                        case 1:
//                            newCol = ofColor(255,0,255);
//                            break;
//                        case 2:
//                            newCol = ofColor(255,255,0);
//                            break;
//                        case 3:
//                            newCol = ofColor(0,0,0);;
//                            break;
//                        default:
//                            newCol = ofColor(255,255,255);
//                            break;
//                    }
//                    pix.setColor(i, j, newCol);
//                    int quant_errorR = oldCol.r-newCol.r;
//                    int quant_errorG = oldCol.g-newCol.g;
//                    int quant_errorB = oldCol.b-newCol.b;
//                    float newVals[3];
//
//                    newVals[0] = pix.getColor(i+1,j).r + weight[0] * quant_errorR;
//                    newVals[1] = pix.getColor(i+1,j).g + weight[0] * quant_errorG;
//                    newVals[2] = pix.getColor(i+1,j).b + weight[0] * quant_errorB;
//                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
//                    pix.setColor(i+1,j, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i+1,j).a));
//                    newVals[0] = pix.getColor(i-1,j+1).r + weight[1] * quant_errorR;
//                    newVals[1] = pix.getColor(i-1,j+1).g + weight[1] * quant_errorG;
//                    newVals[2] = pix.getColor(i-1,j+1).b + weight[1] * quant_errorB;
//                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
//                    pix.setColor(i-1,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i-1,j+1).a));
//                    newVals[0] = pix.getColor(i,j+1).r + weight[2] * quant_errorR;
//                    newVals[1] = pix.getColor(i,j+1).g + weight[2] * quant_errorG;
//                    newVals[2] = pix.getColor(i,j+1).b + weight[2] * quant_errorB;
//                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
//                    pix.setColor(i,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i,j+1).a));
//                    newVals[0] = pix.getColor(i+1,j+1).r + weight[3] * quant_errorR;
//                    newVals[1] = pix.getColor(i+1,j+1).g + weight[3] * quant_errorG;
//                    newVals[2] = pix.getColor(i+1,j+1).b + weight[3] * quant_errorB;
//                    for(int v=0; v<3; v++){ if(newVals[v]>255){newVals[v]=255;}else if(newVals[v]<0){newVals[v]=0;} }
//                    pix.setColor(i+1,j+1, ofColor(round(newVals[0]),round(newVals[1]),round(newVals[2]),pix.getColor(i+1,j+1).a));
//                }
//            }
//        }
        
        resImg.clear();
        resImg.allocate(w, h, OF_IMAGE_COLOR_ALPHA);
        resImg.setFromPixels(pix);
        resImg.save(name+"_"+ofToString(currentLayer)+".png");
//        if(currentLayer==voxelLayer.size()-1){
//            exporting=false;
//            cout << "time: " << ofGetElapsedTimef()-elTime << endl;
//        }else{
//            currentLayer++;
//        }
        currentLayer++;
    }else{
        if(exporting) cout << "time: " << ofGetElapsedTimef()-elTime << endl;
        exporting=false;
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(200);
    if (exporting) {
        ofSetColor(255);
        
        ofTexture tex;
        tex.loadData(pix);
        tex.draw(20, 20, dist.y*2, dist.z*2);
        
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
        name = ofSystemTextBoxDialog("save?","result/dog");
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
