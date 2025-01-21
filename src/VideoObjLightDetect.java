import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.rmi.server.ServerCloneException;
import java.util.ArrayList;

public class VideoObjLightDetect {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        Mat dst = new Mat();
        Mat blurFrame = new Mat();
        Mat grayFrame = new Mat();
        Mat thresholdFrame = new Mat();
        Mat morphFrame = new Mat();

//        VideoCapture capture = new VideoCapture(0,Videoio.CAP_DSHOW);
//        VideoCapture capture = new VideoCapture(0,Videoio.CAP_FFMPEG);
        VideoCapture capture = new VideoCapture(0);
        if(!capture.isOpened()) return;

        // 2^x // fix exposure // EXP_TIME = 2^(-EXP_VAL) // 0 to -13
//        capture.set(Videoio.CAP_PROP_EXPOSURE, -4);
        capture.set(Videoio.CAP_PROP_AUTO_EXPOSURE, 1);

        double frameW = capture.get(Videoio.CAP_PROP_FRAME_WIDTH);
        double frameH = capture.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        double fps = capture.get(Videoio.CAP_PROP_FPS);
        double exposure = capture.get(Videoio.CAP_PROP_EXPOSURE);
        double autoexposure = capture.get(Videoio.CAP_PROP_AUTO_EXPOSURE);

        System.out.println(frameW + "x" + frameH);
        System.out.println("fps "+fps);
        System.out.println("expo"+exposure);
        System.out.println("autoexpo"+autoexposure);

        int key = 0;

        while (key != 27 && key != 'Q'){ //ESC

            if(!capture.read(src)) break;

            Imgproc.cvtColor(src,grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.blur(grayFrame,blurFrame,new Size(9,9));
            Imgproc.threshold(blurFrame,thresholdFrame,240, 255, Imgproc.THRESH_BINARY);

            Mat dilateElem = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(25,25));
            Mat erodeElem = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(25,25));

            Imgproc.erode(thresholdFrame, morphFrame, erodeElem);
            Imgproc.dilate(morphFrame, morphFrame, dilateElem);

            // contour detection
            Mat hierarchy = new Mat();
            ArrayList<MatOfPoint> counters = new ArrayList<>();
            Imgproc.findContours(morphFrame,counters,hierarchy,Imgproc.RETR_CCOMP,Imgproc.CHAIN_APPROX_SIMPLE);

            Imgproc.drawContours(src, counters,-1, new Scalar(128,0,0), 2);

            // find bonding rectangles
            if(counters.size() > 0){
                for (int i = 0; i < counters.size(); i++) {
                    Rect rect = Imgproc.boundingRect(counters.get(i));
                    Imgproc.rectangle(src,rect,new Scalar(0,128,0),2);
                }
            }

            // find the biggest object
            if(counters.size() > 0) {
                double maxArea = 0;
                int maxAreaIndex = 0;

                for (int i = 0; i < counters.size(); i++) {
                    double area = Imgproc.contourArea(counters.get(i));
                    if(area > maxArea){
                        maxArea = area;
                        maxAreaIndex = i;
                    }
                }
                Rect rect = Imgproc.boundingRect(counters.get(maxAreaIndex));
                Imgproc.rectangle(src, rect, new Scalar(0,0,128),2);
            }

            HighGui.imshow("Src", src);
            HighGui.imshow("GrayFrame", grayFrame);
            HighGui.imshow("BlurFrame", blurFrame);
            HighGui.imshow("ThresholdFrame", thresholdFrame);
            HighGui.imshow("MorphFrame", morphFrame);

            key = HighGui.waitKey(20);
        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
