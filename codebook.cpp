#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

#define CHANNELS 3
// 设置处理的图像通道数,要求小于等于图像本身的通道数

///////////////////////////////////////////////////////////////////////////
// 下面为码本码元的数据结构
// 处理图像时每个像素对应一个码本,每个码本中可有若干个码元
// 当涉及一个新领域,通常会遇到一些奇怪的名词,不要被这些名词吓坏,其实思路都是简单的
typedef struct ce {
    uchar   learnHigh[CHANNELS];    // High side threshold for learning
    // 此码元各通道的阀值上限(学习界限)
    uchar   learnLow[CHANNELS];     // Low side threshold for learning
    // 此码元各通道的阀值下限
    // 学习过程中如果一个新像素各通道值x[i],均有 learnLow[i]<=x[i]<=learnHigh[i],则该像素可合并于此码元
    uchar   max[CHANNELS];          // High side of box boundary
    // 属于此码元的像素中各通道的最大值
    uchar   min[CHANNELS];          // Low side of box boundary
    // 属于此码元的像素中各通道的最小值
    int     t_last_update;          // This is book keeping to allow us to kill stale entries
    // 此码元最后一次更新的时间,每一帧为一个单位时间,用于计算stale
    int     stale;                  // max negative run (biggest period of inactivity)
    // 此码元最长不更新时间,用于删除规定时间不更新的码元,精简码本
} code_element;                     // 码元的数据结构

typedef struct code_book {
    code_element    **cb;
    // 码元的二维指针,理解为指向码元指针数组的指针,使得添加码元时不需要来回复制码元,只需要简单的指针赋值即可
    int             numEntries;
    // 此码本中码元的数目
    int             t;              // count every access
    // 此码本现在的时间,一帧为一个时间单位
} codeBook;                         // 码本的数据结构

//-----------------------------------全函数声明-----------------------------------------
//		描述：声明全局变量
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output);
bool addNewPoints();
bool acceptTrackedPoint(int i);

//-----------------------------------全局变量声明-----------------------------------------
//		描述：声明全局变量
//-------------------------------------------------------------------------------------------------
string window_name = "optical flow tracking";
Mat gray;	// 当前图片
Mat gray_prev;	// 预测图片
vector<Point2f> points[2];	// point0为特征点原来的位置，point1为特征点新位置
vector<Point2f> initial;	// 初始化跟踪点的位置
vector<Point2f> features;	// 检测的特征
int maxCount = 500;	// 检测的最大特征数
double qLevel = 0.01;	// 特征检测的等级
double minDist = 10.0;	// 两特征点之间的最小距离
vector<uchar> status;	// 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;
vector<CvRect> roiRect;
vector<Mat> ROI;

//---------------------------opticalflow_part------------------------------------------------------
// function: tracking
// brief: 跟踪
// parameter: frame	输入的视频帧
//			  output 有跟踪结果的视频帧
// return: void
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output)
{

    //
    cvtColor(frame, gray, COLOR_BGR2GRAY);  //转化为灰度图
    //
    //cvtColor(frame, gray, CV_BGR2GRAY);
    for(auto rR:roiRect)
    {
        Mat roi=gray(Rect((rR).x,(rR).y,(rR).width,(rR).height));
        ROI.push_back(roi);
    };

    frame.copyTo(output);

    // 添加特征点
    if (addNewPoints())
    {
        goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist,ROI);  //(输入图像,保存检测出的角点，角点数目最大值，品质因子，)
        points[0].insert(points[0].end(), features.begin(), features.end());
        initial.insert(initial.end(), features.begin(), features.end());
    }

    if (gray_prev.empty())
    {
        gray.copyTo(gray_prev);
    }
    // l-k光流法运动估计
    calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
    // 去掉一些不好的特征点
    int k = 0;
    for (size_t i=0; i<points[1].size(); i++)
    {
        if (acceptTrackedPoint(i))
        {
            initial[k] = initial[i];
            points[1][k++] = points[1][i];
        }
    }
    points[1].resize(k);
    initial.resize(k);
    // 显示特征点和运动轨迹
    for (size_t i=0; i<points[1].size(); i++)
    {
        line(output, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
    }

    // 把当前跟踪结果作为下一参考
    swap(points[1], points[0]);
    swap(gray_prev, gray);

    imshow(window_name, output);
}

//-------------------------------------------------------------------------------------------------
// function: addNewPoints
// brief: 检测新点是否应该被添加
// parameter:
// return: 是否被添加标志
//-------------------------------------------------------------------------------------------------
bool addNewPoints()
{
    return points[0].size() <= 10;
}

//-------------------------------------------------------------------------------------------------
// function: acceptTrackedPoint
// brief: 决定哪些点跟踪点被接受
// parameter:
// return:
//-------------------------------------------------------------------------------------------------
bool acceptTrackedPoint(int i)
{
    return status[i] && ((abs(points[0][i].x - points[1][i].x) + abs(points[0][i].y - points[1][i].y)) > 2);
}


//////////////////////////////////////codebook_part///////////////////////////////////////
// int updateCodeBook(uchar *p, codeBook &c, unsigned cbBounds)
// Updates the codebook entry with a new data point
//
// p            Pointer to a YUV pixel
// c            Codebook for this pixel
// cbBounds     Learning bounds for codebook (Rule of thumb: 10)
// numChannels  Number of color channels we're learning
//
// NOTES:
//      cvBounds must be of size cvBounds[numChannels]
//
// RETURN
//  codebook index
int cvupdateCodeBook(uchar *p, codeBook &c, unsigned *cbBounds, int numChannels)
{
    if(c.numEntries == 0) c.t = 0;
    // 码本中码元为零时初始化时间为0
    c.t += 1;   // Record learning event
    // 每调用一次加一,即每一帧图像加一

    //SET HIGH AND LOW BOUNDS
    int n;
    unsigned int high[3],low[3];
    for (n=0; n<numChannels; n++)
    {
        high[n] = *(p+n) + *(cbBounds+n);
        // *(p+n) 和 p[n] 结果等价,经试验*(p+n) 速度更快
        if(high[n] > 255) high[n] = 255;
        low[n] = *(p+n)-*(cbBounds+n);
        if(low[n] < 0) low[n] = 0;
        // 用p 所指像素通道数据,加减cbBonds中数值,作为此像素阀值的上下限
    }

    //SEE IF THIS FITS AN EXISTING CODEWORD
    int matchChannel;
    int i;
    for (i=0; i<c.numEntries; i++)
    {
        // 遍历此码本每个码元,测试p像素是否满足其中之一
        matchChannel = 0;
        for (n=0; n<numChannels; n++)
            //遍历每个通道
        {
            if((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n])) //Found an entry for this channel
                // 如果p 像素通道数据在该码元阀值上下限之间
            {
                matchChannel++;
            }
        }
        if (matchChannel == numChannels)        // If an entry was found over all channels
            // 如果p 像素各通道都满足上面条件
        {
            c.cb[i]->t_last_update = c.t;
            // 更新该码元时间为当前时间
            // adjust this codeword for the first channel
            for (n=0; n<numChannels; n++)
                //调整该码元各通道最大最小值
            {
                if (c.cb[i]->max[n] < *(p+n))
                    c.cb[i]->max[n] = *(p+n);
                else if (c.cb[i]->min[n] > *(p+n))
                    c.cb[i]->min[n] = *(p+n);
            }
            break;
        }
    }

    // ENTER A NEW CODE WORD IF NEEDED
    if(i == c.numEntries)  // No existing code word found, make a new one
        // p 像素不满足此码本中任何一个码元,下面创建一个新码元
    {
        code_element **foo = new code_element* [c.numEntries+1];
        // 申请c.numEntries+1 个指向码元的指针
        for(int ii=0; ii<c.numEntries; ii++)
            // 将前c.numEntries 个指针指向已存在的每个码元
            foo[ii] = c.cb[ii];

        foo[c.numEntries] = new code_element;
        // 申请一个新的码元
        if(c.numEntries) delete [] c.cb;
        // 删除c.cb 指针数组
        c.cb = foo;
        // 把foo 头指针赋给c.cb
        for(n=0; n<numChannels; n++)
            // 更新新码元各通道数据
        {
            c.cb[c.numEntries]->learnHigh[n] = high[n];
            c.cb[c.numEntries]->learnLow[n] = low[n];
            c.cb[c.numEntries]->max[n] = *(p+n);
            c.cb[c.numEntries]->min[n] = *(p+n);
        }
        c.cb[c.numEntries]->t_last_update = c.t;
        c.cb[c.numEntries]->stale = 0;
        c.numEntries += 1;
    }

    // OVERHEAD TO TRACK POTENTIAL STALE ENTRIES
    for(int s=0; s<c.numEntries; s++)
    {
        // This garbage is to track which codebook entries are going stale
        int negRun = c.t - c.cb[s]->t_last_update;
        // 计算该码元的不更新时间
        if(c.cb[s]->stale < negRun)
            c.cb[s]->stale = negRun;
    }

    // SLOWLY ADJUST LEARNING BOUNDS
    for(n=0; n<numChannels; n++)
        // 如果像素通道数据在高低阀值范围内,但在码元阀值之外,则缓慢调整此码元学习界限
    {
        if(c.cb[i]->learnHigh[n] < high[n])
            c.cb[i]->learnHigh[n] += 1;
        if(c.cb[i]->learnLow[n] > low[n])
            c.cb[i]->learnLow[n] -= 1;
    }

    return(i);
}

///////////////////////////////////////////////////////////////////////////////////
// uchar cvbackgroundDiff(uchar *p, codeBook &c, int minMod, int maxMod)
// Given a pixel and a code book, determine if the pixel is covered by the codebook
//
// p        pixel pointer (YUV interleaved)
// c        codebook reference
// numChannels  Number of channels we are testing
// maxMod   Add this (possibly negative) number onto max level when code_element determining if new pixel is foreground
// minMod   Subract this (possible negative) number from min level code_element when determining if pixel is foreground
//
// NOTES:
// minMod and maxMod must have length numChannels, e.g. 3 channels => minMod[3], maxMod[3].
//
// Return
// 0 => background, 255 => foreground
uchar cvbackgroundDiff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod)
{
    // 下面步骤和背景学习中查找码元如出一辙
    int matchChannel;
    //SEE IF THIS FITS AN EXISTING CODEWORD
    int i;
    for (i=0; i<c.numEntries; i++)
    {
        matchChannel = 0;
        for (int n=0; n<numChannels; n++)
        {
            if ((c.cb[i]->min[n] - minMod[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->max[n] + maxMod[n]))
                matchChannel++; //Found an entry for this channel
            else
                break;
        }
        if (matchChannel == numChannels)
            break; //Found an entry that matched all channels
    }
    if(i == c.numEntries)
        // p像素各通道值满足码本中其中一个码元,则返回白色
        return(255);

    return(0);
}


//UTILITES/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
//int clearStaleEntries(codeBook &c)
// After you've learned for some period of time, periodically call this to clear out stale codebook entries
//
//c     Codebook to clean up
//
// Return
// number of entries cleared
int cvclearStaleEntries(codeBook &c)
{
    int staleThresh = c.t >> 1;         // 设定刷新时间
    int *keep = new int [c.numEntries]; // 申请一个标记数组
    int keepCnt = 0;                    // 记录不删除码元数目
    //SEE WHICH CODEBOOK ENTRIES ARE TOO STALE
    for (int i=0; i<c.numEntries; i++)
        // 遍历码本中每个码元
    {
        if (c.cb[i]->stale > staleThresh)
            // 如码元中的不更新时间大于设定的刷新时间,则标记为删除
            keep[i] = 0; //Mark for destruction
        else
        {
            keep[i] = 1; //Mark to keep
            keepCnt += 1;
        }
    }

    // KEEP ONLY THE GOOD
    c.t = 0;                        //Full reset on stale tracking
    // 码本时间清零
    code_element **foo = new code_element* [keepCnt];
    // 申请大小为keepCnt 的码元指针数组
    int k=0;
    for(int ii=0; ii<c.numEntries; ii++)
    {
        if(keep[ii])
        {
            foo[k] = c.cb[ii];
            foo[k]->stale = 0;      //We have to refresh these entries for next clearStale
            foo[k]->t_last_update = 0;
            k++;
        }
    }
    //CLEAN UP
    delete [] keep;
    delete [] c.cb;
    c.cb = foo;
    // 把foo 头指针地址赋给c.cb
    int numCleared = c.numEntries - keepCnt;
    // 被清理的码元个数
    c.numEntries = keepCnt;
    // 剩余的码元地址
    return(numCleared);
}


// 寻找掩模轮廓
void find_connected_componts(IplImage *raw,IplImage *mask,int ploy_hull0 = 1,float perimScale = 4,CvRect *bbs = NULL,CvPoint *centers = NULL)
{
    static CvMemStorage* mem_storage = NULL;
    static CvSeq* contours = NULL;
    //为寻找轮廓定义存储空间
    /*cvMorphologyEx(mask,mask,0,0,CV_MOP_OPEN,1);
    cvMorphologyEx(mask,mask,0,0,CV_MOP_CLOSE,1); */
    //形态学去除噪声
    if(mem_storage==NULL)
    {
        mem_storage = cvCreateMemStorage(0);
    }
    else
    {
        cvClearMemStorage(mem_storage);
    }
    CvContourScanner scanner = cvStartFindContours(mask,mem_storage,sizeof(CvContour),CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    //寻找外轮廓，保存在scanner中

    CvSeq *c;

    int numCont = 0;
    while((c=cvFindNextContour(scanner))!=NULL)
    {
        double len = cvContourPerimeter(c);
        //该轮廓的长度
        double q = (mask->height+mask->width)/(2*perimScale); //调整噪声滤波阈值使之符合应用场景。
        //所需要轮廓的长度，小于则删除该轮廓
        if (len<q)
        {
            cvSubstituteContour(scanner,NULL);
        }
        else
        {
            CvSeq *c_new;
            c_new = cvApproxPoly(c,sizeof(CvContour),mem_storage,CV_POLY_APPROX_DP,0,0);
            //多边形近似，减少无用点（貌似还可以用凸优化近似）

            cvSubstituteContour(scanner,c_new);
            //替换原轮廓
            numCont++;
        }
    }
    contours = cvEndFindContours(&scanner);

    CvScalar CVX_WHITE = CV_RGB(0xff,0xff,0xff);
    CvScalar CVX_BLACK = CV_RGB(0x00,0x00,0x00);
    //定义绘制轮廓颜色

    cvZero(0);
    //在新的掩模绘制


    CvRect rect;
    int i=0;

    for(i=0,c=contours;c!=NULL&i<10;c=c->h_next,i++)
    {
        bbs[i] = cvBoundingRect(c);
        rect = bbs[i];
        cvDrawContours(mask,c,CVX_WHITE,CVX_WHITE,-1,2,8);
        cvRectangle(raw,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),CVX_WHITE,1,8,0);
        roiRect.push_back(rect);//记得用完过后要释放掉roiRect!!!!
    }
    cvNamedWindow("Raw");
    cvShowImage("Raw", raw);
    cvNamedWindow("CodeBook");
    cvShowImage("CodeBook",mask);
    cvWaitKey(30);
}


int main()
{
    ///////////////////////////////////////
    // 需要使用的变量
    CvCapture*  capture;
    IplImage*   rawImage;
    IplImage*   yuvImage;
    IplImage*   ImaskCodeBook;
    codeBook*   cB;
    CvRect      bbs[20];
    unsigned    cbBounds[CHANNELS];//最开始定义了 CHANNELS=3
    uchar*      pColor; //YUV pointer
    int         imageLen;
    int         nChannels = CHANNELS;
    int         minMod[CHANNELS];
    int         maxMod[CHANNELS];

    //////////////////////////////////////////////////////////////////////////
    // 初始化各变量
    cvNamedWindow("Raw");
    cvNamedWindow("CodeBook");
    //Mat frame;
    Mat result;

    capture = cvCreateCameraCapture(0);
    if (!capture)
    {
        printf("Couldn't open the capture!");
        return -1;
    }
    rawImage = cvQueryFrame(capture);  //从摄像头或者文件抓取一帧，然后解压并返回这一帧。
    Mat frame=cvarrToMat(rawImage,true);
    yuvImage = cvCreateImage(cvGetSize(rawImage), 8, 3);
    // 给yuvImage 分配一个和rawImage 尺寸相同,8位3通道图像
    ImaskCodeBook = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
    // 为ImaskCodeBook 分配一个和rawImage 尺寸相同,8位单通道图像
    cvSet(ImaskCodeBook, cvScalar(255));
    // 设置单通道数组所有元素为255,即初始化为白色图像

    imageLen = rawImage->width * rawImage->height;
    cB = new codeBook[imageLen];
    // 得到与图像像素数目长度一样的一组码本,以便对每个像素进行处理

    for (int i=0; i<imageLen; i++)
        // 初始化每个码元数目为0，即遍历码本，初始化码元。
        cB[i].numEntries = 0;
    for (int i=0; i<nChannels; i++)
    {
        cbBounds[i] = 10;   // 用于确定码元各通道的阀值

        minMod[i]   = 10;   // 用于背景差分函数中
        maxMod[i]   = 10;   // 调整其值以达到最好的分割
    }


    //////////////////////////////////////////////////////////////////////////
    // 开始处理视频每一帧图像
    for (int i=0;;i++)
    {
        cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);
        // 色彩空间转换,将rawImage 转换到YUV色彩空间,输出到yuvImage
        // 即使不转换效果依然很好
        // yuvImage = cvCloneImage(rawImage);

        if (i <= 100)
            // 100帧内进行背景学习
        {
            pColor = (uchar *)(yuvImage->imageData);
            // 指向yuvImage 图像的通道数据
            for (int c=0; c<imageLen; c++)
            {
                cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels); //训练一个基本的背景模型
                // 对每个像素,调用此函数,捕捉背景中相关变化图像
                pColor += 3;
                // 3 通道图像, 指向下一个像素通道数据
            }
            if (i == 100)
                // 到100 帧时调用下面函数,删除码本中陈旧的码元
            {
                for (int c=0; c<imageLen; c++)
                    cvclearStaleEntries(cB[c]); //清楚stale索引。
            }
        }
        else
        {
            uchar maskPixelCodeBook;
            pColor = (uchar *)((yuvImage)->imageData); //3 channel yuv image
            uchar *pMask = (uchar *)((ImaskCodeBook)->imageData); //1 channel image
            // 指向ImaskCodeBook 通道数据序列的首元素
            for(int c=0; c<imageLen; c++)
            {
                maskPixelCodeBook = cvbackgroundDiff(pColor, cB[c], nChannels, minMod, maxMod); //使用训练好的模型分割前景
                *pMask++ = maskPixelCodeBook;
                pColor += 3;
                // pColor 指向的是3通道图像
            }
        }

        find_connected_componts(rawImage,ImaskCodeBook,1,4.0,bbs);
        ////////////////////////////
        /////////opticalflow////////
        if(!frame.empty())
        {
            tracking(frame, result);    // tracking()在下面定义
        }
        else
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        int c = waitKey(50);
        if( (char)c == 27 )
        {
            break;
        }

        if (!(rawImage = cvQueryFrame(capture)))
            break;
        roiRect.clear();
        //cvShowImage("Raw",rawImage);
        //cvShowImage("CodeBook", ImaskCodeBook);
        //cvWaitKey(30);
    }

    cvReleaseCapture(&capture);
    if (yuvImage)
        cvReleaseImage(&yuvImage);
    if(ImaskCodeBook)
        cvReleaseImage(&ImaskCodeBook);
    cvDestroyAllWindows();
    delete [] cB;

    return 0;
}