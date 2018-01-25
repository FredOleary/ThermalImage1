package com.example.fredoleary.thermalimage1;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

/**
 * Created by fredoleary on 1/24/18.
 */

public class ThermalUtil {
    /*
    Flags for various processing
     */
    private static boolean DISPLAY_CONTOURS =  FALSE;           // Displays the contour lines in red
    private static boolean DISPLAY_CONTOUR_RECTS =  FALSE;      // Displays the contour covering rectangle in blue
    private static boolean DISPLAY_COVER_RECT =  TRUE;          // Displays the union of all contour covering rectangles in blue

    private static final String    TAG = "ThermalUtil";

    /* HSV filter colors. Note these are derived empiracally from paint brush.
     RANGES:
        Paintbrush uses 0-360 range for Hue (H), openCV uses 0-180 range
        Paintbrush uses 0-100 range for Saturation/Value (H/V), openCV uses 0-255 range

     */
    private Scalar low_color = new Scalar(40.0/360*180, 100, 100);
    private  Scalar high_color = new Scalar(50.0/360*180, 255, 255);

    /*
    Minimum width.. (Empirical) - Detected object must be this percentage of the image height
     */
    private static final double minObjectHeightPercent = 55.0;

    public Mat getImage(Context context, int resourceId ){
        Mat image = null;
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://"+R.class.getPackage().getName()+"/" +resourceId);
        try {
            stream = context.getContentResolver().openInputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return image;
        }

        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap bmp = BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
        image = new Mat();
        Utils.bitmapToMat(bmp, image);

        return image;
    }
    public void displayImage(Context context, ImageView imageView, Mat image)
    {
        // create a bitMap
        Bitmap bitMap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        // convert to bitmap:
        Utils.matToBitmap(image, bitMap);

        int width = context.getResources().getDisplayMetrics().widthPixels;
        int height = (width*bitMap.getHeight())/bitMap.getWidth();
        Bitmap bitmapScaled = Bitmap.createScaledBitmap(bitMap, width, height, true);

        // find the imageview and draw it!
        //ImageView iv = (ImageView) context.findViewById(R.id.imgview);
        imageView.setImageBitmap(bitmapScaled);
    }

    public Mat getMonoChromeImage(Mat image ){
        Mat hsvImage;
        hsvImage = new Mat();
        // Convert to HSV for filtering
        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_RGB2HSV);
        Mat mask = new Mat();

//        Scalar low_color = new Scalar(40.0/360*180, 100, 100);
//        Scalar high_color = new Scalar(50.0/360*180, 255, 255);
        Core.inRange(hsvImage, low_color, high_color, mask);


        // Use Canny to create the contours on the monochrome image
        Mat contourImage = new Mat();
        Imgproc.Canny(mask, contourImage, 0, 40);
        return contourImage;

    }

    public boolean processContours( Mat monoImage, Mat originalImage ){
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(monoImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Rect unionRect = null;
        for (MatOfPoint cnt : contours) {
            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());
            MatOfPoint2f approxCurve = new MatOfPoint2f();


            // approximates a polygonal curve with the specified precision
            Imgproc.approxPolyDP(
                    curve,
                    approxCurve,
                    0.1 * Imgproc.arcLength(curve, true),
                    true
            );

            int numberVertices = (int) approxCurve.total();
            double contourArea = Imgproc.contourArea(cnt);
            Log.d(TAG, "numberVertices: " + numberVertices + ". contourArea: " + contourArea);

            if( DISPLAY_CONTOURS ) {
                List<MatOfPoint> list = new ArrayList<>();
                list.add(cnt);
                Imgproc.drawContours(originalImage, list, 0, new Scalar(255, 0, 0), 3);
            }
            Rect r = Imgproc.boundingRect(cnt);

            /* Filter contours... */
            if( r.area() >  1000 ) {
                if( unionRect == null){
                    unionRect = r;
                }else{
                    unionRect = unionRect( unionRect, r);
                }
                Log.d(TAG, "Rectangle area: " + r.area());

                if(DISPLAY_CONTOUR_RECTS) {
                    Imgproc.rectangle(
                            originalImage,
                            new Point(r.x, r.y),
                            new Point(r.x + r.width, r.y + r.height),
                            new Scalar(0, 0, 255),
                            2);
                }
            }

        }

        if( unionRect != null ){
            double imageWidthPct = (double)unionRect.width/(double)monoImage.width()*100;
            double imageHeightPct = (double)unionRect.height/(double)monoImage.height()*100;
            Log.d(TAG, "COVER WIDTH: " + imageWidthPct + "%. COVER HEIGHT: " + imageHeightPct + "%");
            if( imageHeightPct > minObjectHeightPercent ) {
                if (DISPLAY_COVER_RECT) {
                    Imgproc.rectangle(
                            originalImage,
                            new Point(unionRect.x, unionRect.y),
                            new Point(unionRect.x + unionRect.width, unionRect.y + unionRect.height),
                            new Scalar(0, 0, 255),
                            2);
                }
                return true;
            }
        }
        return false;
    }
    private Rect unionRect( Rect r1, Rect r2 ){
        int x, y, w, h;
        x = Math.min(r1.x, r2.x);
        y = Math.min(r1.y, r2.y);
        w = Math.max(r1.x+r1.width, r2.x+r2.width) - x;
        h = Math.max(r1.y+r1.height, r2.y+r2.height) - y;
        return new Rect(x, y, w, h);

    }

}
