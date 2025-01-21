import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;

public class MorphologyOnImage {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = Imgcodecs.imread("./test_images/eight.bmp");
        Mat dst = new Mat();
        Mat grayFrame = new Mat();
        Mat thresholdFrame = new Mat();
        Mat morphFrame = new Mat();
        Mat extractFrame = new Mat();


        Imgproc.cvtColor(src, grayFrame, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.blur(grayFrame, grayFrame, new Size(19, 19));
        Imgproc.threshold(grayFrame, thresholdFrame, 130, 255, Imgproc.THRESH_BINARY_INV); //100 120

        Mat dilateElem = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(25, 25));
        Mat erodeElem = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(23, 23));

        System.out.println(dilateElem.dump());
        System.out.println(CvType.typeToString(dilateElem.type()));
        HighGui.imshow("Dilate Elem", dilateElem);
        HighGui.imshow("Erode Elem", erodeElem);

        Imgproc.dilate(thresholdFrame, morphFrame, dilateElem);
        Imgproc.erode(morphFrame, morphFrame, erodeElem);

//        Imgproc.morphologyEx(thresholdFrame,morphFrame,Imgproc.MORPH_CLOSE,dilateElem);
//        Imgproc.morphologyEx(thresholdFrame,morphFrame,Imgproc.MORPH_OPEN,dilateElem);

        Core.copyTo(src,extractFrame,morphFrame);

        dst = src.clone();

        // find Contours
        Mat hierarchy = new Mat();
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(morphFrame, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        Imgproc.drawContours(dst, contours, -1, new Scalar(255, 0, 0), 2);
        Imgproc.putText(dst,"Objects: "+contours.size(),new Point(0,dst.height()-5),Imgproc.FONT_HERSHEY_PLAIN,1.5,new Scalar(150,0,0),2);

        // find Bounding Rectangles
        if (contours.size() > 0) {
            for (int i = 0; i < contours.size(); i++) {
                Rect rect = Imgproc.boundingRect(contours.get(i));
                Imgproc.rectangle(dst, rect, new Scalar(0, 255, 0), 2);
            }
        }

        // find the biggest object
        if (contours.size() > 0) {
            double maxArea = 0;
            int maxAreaIndex = 0;

            for (int i = 0; i < contours.size(); i++) {
                double area = Imgproc.contourArea(contours.get(i));
                if (area > maxArea) {
                    maxArea = area;
                    maxAreaIndex = i;
                }
            }
            Rect rect = Imgproc.boundingRect(contours.get(maxAreaIndex));
            Imgproc.rectangle(dst, rect, new Scalar(0, 0, 255), 2);
        }

        HighGui.imshow("Src", src);
        HighGui.imshow("Gray", grayFrame);
        HighGui.imshow("Binary", thresholdFrame);
        HighGui.imshow("Morph", morphFrame);
        HighGui.imshow("Extract", extractFrame);
        HighGui.imshow("Dst", dst);

        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
