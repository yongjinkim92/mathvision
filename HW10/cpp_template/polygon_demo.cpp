#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;


PolygonDemo::PolygonDemo()
{
    m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!m_data_ready)
        putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

    
    drawPolygon(frame, m_data_pts, m_data_ready);
    if (m_data_ready)
    {
        // polygon area
        if (m_param.compute_area)
        {
            int area = polyArea(m_data_pts);
            char str[100];
            sprintf_s(str, 100, "Area = %d", area);
            putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
        }
        
        // pt in polygon
        if (m_param.check_ptInPoly)
        {
            for (int i = 0; i < (int)m_test_pts.size(); i++)
            {
                if (ptInPolygon(m_data_pts, m_test_pts[i]))
                {
                    circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                }
                else
                {
                    circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
                }
            }
        }
        

        // homography check
        if (m_param.check_homography && m_data_pts.size() == 4)
        {
            // rect points
            int rect_sz = 100;
            vector<Point> rc_pts;
            rc_pts.push_back(Point(0, 0));
            rc_pts.push_back(Point(0, rect_sz));
            rc_pts.push_back(Point(rect_sz, rect_sz));
            rc_pts.push_back(Point(rect_sz, 0));
            rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

            // draw mapping
            char* abcd[4] = { "A", "B", "C", "D" };
            for (int i = 0; i < 4; i++)
            {
                line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
                circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
            }

            // check homography
            int homo_type = classifyHomography(rc_pts, m_data_pts);
            char type_str[100];
            switch (homo_type)
            {
            case NORMAL:
                sprintf_s(type_str, 100, "normal");
                break;
            case CONCAVE:
                sprintf_s(type_str, 100, "concave");
                break;
            case TWIST:
                sprintf_s(type_str, 100, "twist");
                break;
            case REFLECTION:
                sprintf_s(type_str, 100, "reflection");
                break;
            case CONCAVE_REFLECTION:
                sprintf_s(type_str, 100, "concave reflection");
               break;
            }

            putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }

        // fit circle
        if (m_param.fit_circle)
        {
            Point2d center;
            double radius = 0;
            bool ok = fitCircle(m_data_pts, center, radius);
            if (ok)
            {
                circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
                circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
            }
        }

		// fit ellipse
		if (m_param.fit_ellipse)
		{
			Point2d m;
			Point2d v;
			float theta;
			bool ok = fitEllipse(m_data_pts, m, v, theta);
			if (ok)
			{
				ellipse(frame, m, Size((int)v.x, (int)v.y), theta, 0, 360, Scalar(0,255,0), 1);
				ellipse(frame, m, Size(2, 2), theta, 0, 360, Scalar(0, 255, 0), -1);
			}
		}

        // fit line
        if (m_param.fit_line)
        {
            Point2d pt1, pt2, pt3, pt4;
            
            bool ok = fitLine(m_data_pts, pt1, pt2, pt3, pt4);
            if (ok) {
            // y = ax + b
            line(frame, pt1, pt2, Scalar(0, 255, 0));
            putText(frame, "LS : y = ax + b", Point(10, 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(0, 255, 0));
            // ax + by + c = 0
            line(frame, pt3, pt4, Scalar(0, 0, 255));
            putText(frame, "LS : ax + by + c = 0", Point(10, 40), FONT_HERSHEY_SIMPLEX, .6, Scalar(0, 0, 255));
            }
        }

    }

    imshow("PolygonDemo", frame);
}

// return the area of polygon // HW4_Yongjin kim 
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
    int area = 0;

    for (int i = 1; i < vtx.size() - 1 ; i++)
    {
        area += ((vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y) -
            (vtx[i + 1].x - vtx[0].x) * (vtx[i].y - vtx[0].y)) / 2 ;
    }

    print(vtx);

    return abs(area);
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
    return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION HW4_Yongjin kim 
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
    // pts1은 화면 왼쪽 위 원형모습 좌표, pts2는 내가 클릭한 좌표 
    if (pts1.size() != 4 || pts2.size() != 4) return -1;

    // 도형 판단을 위해 외적 
    float CrossProduct[4];

    CrossProduct[0] = (pts2[0].x - pts2[1].x) * (pts2[2].y - pts2[1].y) -
        (pts2[2].x - pts2[1].x) * (pts2[0].y - pts2[1].y);
    CrossProduct[1] = (pts2[1].x - pts2[2].x) * (pts2[3].y - pts2[2].y) -
        (pts2[3].x - pts2[2].x) * (pts2[1].y - pts2[2].y);
    CrossProduct[2] = (pts2[2].x - pts2[3].x) * (pts2[0].y - pts2[3].y) -
        (pts2[0].x - pts2[3].x) * (pts2[2].y - pts2[3].y);
    CrossProduct[3] = (pts2[3].x - pts2[0].x) * (pts2[1].y - pts2[0].y) -
        (pts2[1].x - pts2[0].x) * (pts2[3].y - pts2[0].y);


    //외적의 양수 음수 개수로 도형판단
    int pos = 0;
    for (int i = 0; i < pts2.size(); i++)
    {
        if (CrossProduct[i] > 0)
            pos++;
    }
    if (pos == 0)
        return REFLECTION;
    else if (pos == 4)
        return NORMAL;
    else if (pos == 3)
        return CONCAVE;
    else if (pos == 2)
        return TWIST;
    else if (pos == 1)
        return CONCAVE_REFLECTION;

 
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
    int n = (int)pts.size();
    if (n < 3) return false;

	Mat M = Mat::zeros(n, 3, CV_32F);
	Mat R = Mat(n, 1, CV_32F);

	for (int i = 0; i < n; i++) 
	{
		int x = pts[i].x;
		int y = pts[i].y;

		M.at<float>(i, 0) = x;
		M.at<float>(i, 1) = y;
		M.at<float>(i, 2) = 1;

		R.at<float>(i, 0) = -(pow(x, 2) + pow(y, 2));
	}
	
	Mat M_T = M.t();

	Mat M_PInv = ((M_T * M).inv())*(M_T);

	Mat X = M_PInv * R;
	

	float a = X.at<float>(0, 0);
	float b = X.at<float>(1, 0);
	float c = X.at<float>(2, 0);

	
	center.x = -a / 2;
	center.y = -b / 2;
	radius = sqrt(pow(a, 2) + pow(b, 2) - 4 * c) / 2.;

    return true;
}

bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& m, cv::Point2d& v, float& theta)
{
	int n = (int)pts.size();

	if (n < 4) return false;

	Mat M = Mat::zeros(n, 6, CV_32F);

	for (int i = 0; i < n; i++)
	{
		int x = pts[i].x;
		int y = pts[i].y;

		M.at<float>(i, 0) = pow(x,2);
		M.at<float>(i, 1) = x*y;
		M.at<float>(i, 2) = pow(y,2);
		M.at<float>(i, 3) = x;
		M.at<float>(i, 4) = y;
		M.at<float>(i, 5) = 1;
	}

	Mat w;
	Mat u;
	Mat v_t;

	SVD::compute(M, w, u, v_t, SVD::FULL_UV);

	Mat X = v_t.row(v_t.rows - 1);

	float a = X.at<float>(0, 0);
	float b = X.at<float>(0, 1);
	float c = X.at<float>(0, 2);
	float d = X.at<float>(0, 3);
	float e = X.at<float>(0, 4);
	float f = X.at<float>(0, 5);

	m.x = ((2 * c * d) - (b*e)) / (pow(b, 2) - (4 * a*c));
	m.y = ((2 * a * e) - (b*d)) / (pow(b, 2) - (4 * a*c));

	theta = (1./2.) * atan2(b, a - c);

	float ap = (a * pow(cos(theta), 2) + b * cos(theta) * sin(theta) + c * pow(sin(theta), 2));
	float cp = (a * pow(sin(theta), 2) - b * cos(theta) * sin(theta) + c * pow(cos(theta), 2));

	float scale_inv = (a * pow(m.x, 2) + b * m.x * m.y + c * pow(m.y, 2) - f);

	v.x = sqrt(scale_inv / ap);
	v.y = sqrt(scale_inv / cp);

	theta = theta * (180 / 3.141592653);

	Mat R = M * X.t();

	//cout << "center: (" << m.x << ", " << m.y << ")" << endl;
	//cout << "major axis(X_axis): " << v.x << endl;
	//cout << "minor axis(Y_axis): " << v.y << endl;
	//cout << "theta: " << theta << endl;
	cout << "Ax: " << R.t() << endl;
	//cout << "Left Singular vector: " << u << endl;
	//cout << "Singular Value: " << w << endl;
	//cout << "right Singular vector: " << v_t << endl;
	//cout << "X: " << X << endl;

	return true;

}

bool PolygonDemo::fitLine(const std::vector<cv::Point>& pts, cv::Point2d& pt1, cv::Point2d& pt2, cv::Point2d& pt3, cv::Point2d& pt4)
{
    int n = (int)pts.size();

    if (n < 2) return false;

    // y = ax + b

    Mat A = Mat::ones(n, 2, CV_32F);
    Mat b = Mat(n, 1, CV_32F);

    for (int i = 0; i < n; i++)
    {
        int x = pts[i].x;
        int y = pts[i].y;
        A.at<float>(i, 0) = x;
        b.at<float>(i, 0) = y;
    }

    Mat A_pinv;
    invert(A, A_pinv, DECOMP_SVD);
    Mat x = A_pinv * b;
    
    pt1.x = 0;
    pt1.y = x.at<float>(0, 0) * pt1.x + x.at<float>(1, 0);

    pt2.x = 640;
    pt2.y = x.at<float>(0, 0) * pt2.x + x.at<float>(1, 0);

    // ax + by + c = 0
    Mat A_p = Mat::ones(n, 3, CV_32F);
    Mat b_p = Mat::zeros(n, 1, CV_32F);

    for (int i = 0; i < n; i++) {
        int x = pts[i].x;
        int y = pts[i].y;

        A_p.at<float>(i, 0) = x;
        A_p.at<float>(i, 1) = y;
    }

    Mat w, u, vt;

    SVD::compute(A_p, w, u, vt, SVD::FULL_UV);

    
    Mat x_p = vt.row(vt.rows - 1);

    pt3.x = 0;
    pt3.y = -((x_p.at<float>(0, 0) * pt3.x + x_p.at<float>(0, 2)) / x_p.at<float>(0, 1));

    pt4.x = 640;
    pt4.y = -((x_p.at<float>(0, 0) * pt4.x + x_p.at<float>(0, 2)) / x_p.at<float>(0, 1));

    return true;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
    }
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
        if (!m_data_ready)
        {
            m_data_pts.push_back(Point(x, y));
        }
        else
        {
            m_test_pts.push_back(Point(x, y));
        }
        refreshWindow();
    }
    else if (evt == CV_EVENT_LBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_LBUTTONDBLCLK)
    {
        m_data_ready = true;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONDBLCLK)
    {
    }
    else if (evt == CV_EVENT_MOUSEMOVE)
    {
    }
    else if (evt == CV_EVENT_RBUTTONDOWN)
    {
        m_data_pts.clear();
        m_test_pts.clear();
        m_data_ready = false;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_MBUTTONDOWN)
    {
    }
    else if (evt == CV_EVENT_MBUTTONUP)
    {
    }

    if (flags&CV_EVENT_FLAG_LBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_RBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_MBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_CTRLKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_SHIFTKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_ALTKEY)
    {
    }
}
