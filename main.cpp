
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include <unordered_map>
#include <string>
#include "ViewPair.h"
#include "Mesh.h"
#include <opencv2/opencv.hpp>

#include "SparseCloud.h"
#include "pba/pba.h"

using namespace std;

vector<cv::Point2f> im1kp,im2kp;


int currentW = 800;
int currentH = 600;

int targetFramerate = 60;

int mouseLastX, mouseLastY;
float modelRotX = 30, modelRotZ = 10, viewRotX = 0, viewRotY = 0, viewDist = 3;

int windowId;
bool displayTriangles = true;
bool displayPoints = true;

struct keyFunction {
	const string description;
	void(*function)(void);
};
unordered_map<char, keyFunction> keys = {
	{ 27,
	{ "Exits the program", [](void) {
		glutDestroyWindow(windowId);
		exit(EXIT_SUCCESS);
	} }
	},
	{ 't',
	{ "Switches triangles' display", [](void) {
		displayTriangles = !displayTriangles;
	} }
	},
	{ 'p',
	{ "Switches the point cloud's display", [](void) {
		displayPoints = !displayPoints;
	} }
	}
};


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event ==cv::EVENT_LBUTTONDOWN)
	{
		//cout << "LEFT button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		im1kp.push_back(cv::Point2f(x, y));
		cout << "size kp1 " << im1kp.size() << endl;
	}
	if (event == cv::EVENT_RBUTTONDOWN)
	{
		//cout << "RIGHT button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		im2kp.push_back(cv::Point2f(x, y));
		cout << "size kp2 " << im2kp.size() << endl;
	}
}


Mesh mesh;

void init() {

	glClearColor(0.0, 0.0, 0.0, 1.0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// vertices
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, mesh.vertices.data());
	//glPointSize(3);

	// vertice colors
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, mesh.colors.data());
}

void display() {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glRotated(viewRotX, 1, 0, 0);
	glRotated(viewRotY, 0, 1, 0);

	gluLookAt(
		0, -viewDist, 0,
		0, 0, 0,
		0, 0, 1
		);

	glRotated(modelRotZ, 0, 0, 1);
	glRotated(modelRotX, 1, 0, 0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(50, ((double)currentW) / currentH, 0.1, 100);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(6.0);


	if (displayPoints) {
		glDrawArrays(GL_POINTS, 0, mesh.vertices.size() / 3);
	}

	if (displayTriangles) {
		glDrawElements(GL_TRIANGLES, mesh.triangleIndexes.size(), GL_UNSIGNED_INT, mesh.triangleIndexes.data());
	}

	glutSwapBuffers();
}

void idle() {

	static int nWaitUntil = glutGet(GLUT_ELAPSED_TIME);
	int nTimer = glutGet(GLUT_ELAPSED_TIME);
	if (nTimer >= nWaitUntil) {
		glutPostRedisplay();
		nWaitUntil = nTimer + (int)(1000.0f / targetFramerate);
	}
}

void keyboard(unsigned char key, int x, int y) {

	auto function = keys.find(key);
	if (function != keys.end()) {
		function->second.function();
	}

}

void keyboardSpecial(int key, int x, int y) {

}

void reshape(GLsizei w, GLsizei h) {

	currentW = w;
	currentH = h;
	glViewport(0, 0, w, h);
	display();
}

bool lastKeyIsLeft; // left or right

void mouseMove(int x, int y) {

	if (lastKeyIsLeft) {
		modelRotX += 0.2f*(y - mouseLastY);
		modelRotZ += 0.2f*(x - mouseLastX);
		if (modelRotX >= 360) { modelRotX -= 360; }
		if (modelRotZ >= 360) { modelRotZ -= 360; }
	}
	else {
		viewRotX += 0.2f*(y - mouseLastY);
		viewRotY += 0.2f*(x - mouseLastX);
		if (viewRotX >= 360) { viewRotX -= 360; }
		if (viewRotY >= 360) { viewRotY -= 360; }
	}
	mouseLastX = x;
	mouseLastY = y;
}

void mouseClick(int button, int state, int x, int y) {

	if (state == GLUT_DOWN) {

		switch (button)
		{
		case 3: // scroll down
			viewDist -= 0.3f; if (viewDist < 0.1f) { viewDist = 0.1f; }
			break;
		case 4: // scroll up
			viewDist += 0.3f;
			break;
		case 0: // left click
			lastKeyIsLeft = true;
			mouseLastX = x;
			mouseLastY = y;
			break;
		case 2: // right click
			lastKeyIsLeft = false;
			mouseLastX = x;
			mouseLastY = y;
			break;
		}
	}
}

#include <fstream>
#include <sstream>

enum LoadType{
	MANUAL,
	AUTO,
	VSFM
};

int main(int argc = 0, char *argv[] = NULL)
{
	cv::Mat im1 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/roman/0010.jpg");
	cv::Mat im2 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/roman/0011.jpg");
    cv::Mat im3 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/roman/0012.jpg");
    cv::Mat im4 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/roman/0013.jpg");


    /*cv::Mat im1 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/doll/IMG_0998.JPG");
    cv::Mat im2 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/doll/IMG_0999.JPG");
    cv::Mat im3 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/doll/IMG_1000.JPG");
    cv::Mat im4 = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/reconstruction3D/doll/IMG_1001.JPG");*/

    SparseCloud sc;
	sc.initialize(im1,im2);

    sc.addView(im3);
    //sc.addView(im4);
    sc.generateCloud();


    /*
    cv::Ptr<cv::Feature2D> feat2D = cv::xfeatures2d::SIFT::create();
    cv::DescriptorMatcher* bf = new cv::BFMatcher;
    View v1(im1, feat2D);
    View v2(im2, feat2D);
    ViewPair v1v2(&v1, &v2);

    //========= choose how to load matching point ===========
    int nPoint = 30;
    LoadType lt = AUTO;
    switch (lt){
    case AUTO:
        v1v2.matchFeat(bf);
        break;
    case MANUAL:
        cv::namedWindow("im1");
        cv::namedWindow("im2");
        cv::setMouseCallback("im1", CallBackFunc, NULL);
        cv::setMouseCallback("im2", CallBackFunc, NULL);
        cv::imshow("im1", im1);
        cv::imshow("im2", im2);

        cout << "select " << nPoint << "points " << endl;
        while (im1kp.size() != nPoint || im2kp.size() != nPoint){
            cv::waitKey(30);
        }
        v1v2.setManualMatch(im1kp, im2kp);
        break;
    case VSFM:
        Mesh m = Mesh::loadNVM("D:/Pictures/dollNVM.nvm");
        cv::Point2f center(im1.cols / 2, im1.rows / 2);
        //cv::Point2f center(0, 0);
        for (auto f : m.views[0].features){
            im1kp.push_back(cv::Point2f(f.x + center.x, f.y + center.y));
        }
        for (auto f : m.views[1].features){
            im2kp.push_back(cv::Point2f(f.x + center.x, f.y + center.y));
        }
        v1v2.setManualMatch(im1kp, im2kp);
        break;
    }
    */

	//==================== display 3D point cloud =========================
	/*int factor = 1;
	mesh.points = vector<Mesh::Point>(cloud.cols);
	for (int i = 0; i < cloud.cols; i++) {
		mesh.points[i].pos.x =  cloud.at<float>(0,i)*factor;
		mesh.points[i].pos.y = cloud.at<float>(1, i) *factor;
		mesh.points[i].pos.z = cloud.at<float>(2, i) *factor;

		//cv::Vec3b color = im1.at<cv::Vec3b>(im1kp[i].y, im1kp[i].x);
		cv::Vec3b color(255,0, 0);
		mesh.points[i].col.r = color.val[2];
		mesh.points[i].col.g = color.val[1];
		mesh.points[i].col.b = color.val[0];
	}*/


    mesh.points = sc.m_cloud3D;

	mesh.preRender();
	mesh.reCenter();
	mesh.autoScale();
	glutInit(&argc, argv);

	glutInitWindowSize(currentW, currentH);

	windowId = glutCreateWindow("OpenGL window");

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);
	glutReshapeFunc(reshape);
	glutMotionFunc(mouseMove);
	glutMouseFunc(mouseClick);

	glewInit();

	init();

	glutMainLoop();



	return EXIT_SUCCESS;
}