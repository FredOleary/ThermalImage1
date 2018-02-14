package com.example.fredoleary.thermalimage1;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.MORPH_DILATE;
import static org.opencv.imgproc.Imgproc.MORPH_ERODE;


/**
 * Created by fredoleary on 1/24/18.
 */

public class FaceDetectUtil {
    class DetectResult {
        // In order of granularity... (most-least) contour->approxContour->hull->rect
        public boolean isFace;                  // True if the contour could be a face
        public Rect rect;                       // Rectangle that completely covers the contour. Least granular
        public MatOfPoint contour;              // The contour. (Most granular)
        public MatOfPoint approxContour;        // Approximate contour
        public MatOfPoint hull;                 // Polygon cover

    }
    /*
    Flags for various processing
     */


    private static final String    TAG = FaceDetectUtil.class.getName();
    private static boolean DEBUG = true;

    private  static final int MaxThermalRows = 32;
    private  static final int MaxThermalCols = 32;

    private  static final int MatBuffer = 10;

    private  static final int MaxSkinInCBy10 = 400;     // Max temp, degrees (40.0) C by 10
    private  static final int MinSkinInCBy10 = 320;     // Min temp, degrees (32.0) C by 10

    /*
    Minimum size.. (Empirical) - Detected object must be larger than this size. E.g. 10% of the image size
     */
    private static final int minRectSizePct = 10;

    /*
    Maximum size.. (Empirical) - Detected object must be smaller than this size. E.g. 90% of the image size
    */
    private static final int maxRectSizePct = 90;

    /*
    Width/Height ratio. Must be < maxWidthPct and > minWidthPct
     */
    private static final int maxWidthPct =100;
    private static final int minWidthPct =50;

    /*
    Image width cannot exceed maxScreenPct of the screen width
     */
    private static final int maxScreenPct = 90;

    /*
    TODO - Refactor as needed
     */
    public List<DetectResult> detectFaces(Bitmap bitmap, int[][] thermalData) {
        Mat originalImage = new Mat();
        Utils.bitmapToMat(bitmap, originalImage);
        Mat[] maskAndContours = getMonoChromeImageEx(thermalData);
//        Mat monoChromeImage = maskAndContours[0];
        Mat contourImage = maskAndContours[1];
        List<DetectResult> results = processContours(contourImage);
        // TODO - Draw objects on originalImage ???
        if (DEBUG) Log.d(TAG, "hot spots detected=" + results.size());
        return results;
    }

    public Mat getImage(Context context, int resourceId ){
        Mat image = null;
        InputStream stream;
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


    /*
    getMonoChromeImageEx. Produce a monochome image from a 32*32 array of integers where each
    Integer is degrees C * 10. E.g. 32.3 is 323
     */
    public Mat[] getMonoChromeImageEx(int[][] thermalData ){
        Mat[] results= new Mat[2];
        Mat monSource = new Mat(MaxThermalRows + (MatBuffer*2),MaxThermalCols +(MatBuffer*2), CvType.CV_8U);
        initializeFromCentigrade(monSource, thermalData );
        results[0] = monSource;

        int morph_size = 2;
        Mat element = Imgproc.getStructuringElement( Imgproc.MORPH_RECT, new Size( morph_size + 1, morph_size+1 ), new Point( morph_size, morph_size ) );

        Mat morphedMask = new Mat();

        Point kernel = new Point(-1,-1);
        Imgproc.morphologyEx( monSource, morphedMask, MORPH_DILATE, element, kernel, 3);

//        Mat morphedMaskErode = new Mat();
//        Imgproc.morphologyEx( morphedMask, morphedMaskErode, MORPH_ERODE, element, kernel, 1);

        results[0] = morphedMask;

        // Use Canny to create the contours on the monochrome image
        Mat contourImage = new Mat();
        Imgproc.Canny(morphedMask, contourImage, 0, 40);
        results[1] = contourImage;
        return results;
    }
    /*
    Create 32 *32 Mat with white pixels for 'hot' regions.
     */
    private void initializeFromCentigrade( Mat destMat, int[][] thermalData){
        byte[] buffer = new byte[destMat.height()* destMat.width()];
        destMat.put(0,0, buffer);
        for( int rowIndex = 0; rowIndex < MaxThermalRows; rowIndex++ ){
            for( int colIndex = 0; colIndex < MaxThermalRows; colIndex++ ){
                if( thermalData[rowIndex][colIndex] >= MinSkinInCBy10 &&  thermalData[rowIndex][colIndex] <= MaxSkinInCBy10 ){
                    destMat.put(rowIndex+MatBuffer,colIndex+MatBuffer, 0xff );
                }
            }
        }
    }


    public List<DetectResult> processContours( Mat monoImage ) {
        if (DEBUG) Log.d(TAG, "processContours - Begin ------------------------------" );
        List<DetectResult> results = new ArrayList<>();

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(monoImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if (DEBUG) Log.d( TAG, "findContours returned " +  contours.size() + " contours");
        for (MatOfPoint contour : contours) {
            DetectResult result = processContour(contour, monoImage);
            if(result != null ){
                results.add( result);
            }
        }
        if (DEBUG) Log.d(TAG, "processContours - End ------------------------------" );
        return results;
    }

    private DetectResult processContour( MatOfPoint contour, Mat monoImage ){
        DetectResult result = new DetectResult();
        result.isFace = false;
        result.contour = contour;
        Rect boundingRect = Imgproc.boundingRect(contour);
        result.rect = boundingRect;
        int imageSize = monoImage.width() * monoImage.height();

        int rectAreaPct = (int)(boundingRect.area()/(double)imageSize * 100);
        if( rectAreaPct >  minRectSizePct &&  rectAreaPct < maxRectSizePct) {
            if (DEBUG) Log.d(TAG, "Rectangle included. Area%: " + rectAreaPct);

            // approximates a polygonal curve with the specified precision
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(curve, approxCurve, 0.01 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();

            // Require at least 5 vertices... (empirical)
            if( numberVertices > 5){
                if (DEBUG) Log.d(TAG, "Shape included. numberVertices: " + numberVertices);

                MatOfPoint approxCurveInt = new MatOfPoint(approxCurve.toArray());
                result.approxContour = approxCurveInt;

                // Get the hull
                MatOfInt hull = new MatOfInt();
                Imgproc.convexHull(approxCurveInt, hull);

                MatOfPoint hullPoints = hullToMapOfPoint(approxCurveInt, hull );
                result.hull = hullPoints;

                double hullArea = Imgproc.contourArea( hullPoints);
                double polyArea = Imgproc.contourArea( approxCurveInt);
                int areaRatioPct = (int)(polyArea/hullArea *100);


                if( areaRatioPct > 85 ){
                    if (DEBUG) Log.d(TAG, "Shape MATCH: areaRatioPct " + areaRatioPct );

                    // Checking that width fits withing screen. (Too wide is rejected)
                    int screenWidth = monoImage.width();
                    int screenPct = (int)((double)boundingRect.width/(double)screenWidth*100);

                    if(screenPct < maxScreenPct ) {
                        if (DEBUG) Log.d(TAG, "Image width OK: maxScreenPct " + screenPct);

                        // Checking ratio of width to height
                        int widthPct = (int) ((double) boundingRect.width / (double) boundingRect.height * 100);
                        if (widthPct <= maxWidthPct && widthPct >= minWidthPct) {
                            if (DEBUG) Log.d(TAG, "Width/Height MATCH: widthPct " + widthPct);
                            result.isFace = true;
                        } else {
                            if (DEBUG) Log.d(TAG, "Width/Height MISMATCH: widthPct " + widthPct);
                        }
                    }else{
                        if (DEBUG) Log.d(TAG, "Image too WIDE: maxScreenPct " + screenPct);
                    }

                }else{
                    if (DEBUG) Log.d(TAG, "Shape MISMATCH: areaRatioPct " + areaRatioPct );
                }
            }else{
                if (DEBUG) Log.d(TAG, "Contour excluded. Too few numberVertices: " + numberVertices);
            }

        }else{
            if (DEBUG) Log.d(TAG, "Rectangle excluded. Area too small/large, Area%: " + rectAreaPct);
        }
        clipRect( result.rect );
        return result;
    }

    /*
    Clip the bounding rect so that it lies within the 32*32 area. The effect of dilating the monochrome
    image is that some points may lie outside this region
     */
    private void clipRect( Rect rect ){

       if( rect.x < MatBuffer) rect.x = MatBuffer;
       if( rect.y < MatBuffer) rect.y = MatBuffer;
       if( (rect.x - MatBuffer) + rect.width > MaxThermalCols) rect.width = MaxThermalCols - (rect.x - MatBuffer);
       if( (rect.y - MatBuffer) + rect.height > MaxThermalRows) rect.height = MaxThermalRows - (rect.y - MatBuffer);

    }
    public void displayPoly( Mat originalImage, MatOfPoint poly, Scalar color ){
        List<MatOfPoint> polyList = new ArrayList<>();
        Point[] pts = poly.toArray();
        int xyScale = 448/32;
        for( Point pt : pts){
            pt.x = ((pt.x - MatBuffer )* xyScale) + ((640-448)/2);
            pt.y = ((pt.y - MatBuffer )* xyScale) + ((480-448)/2);
        }

        MatOfPoint scaledPoints = new MatOfPoint();
        scaledPoints.fromArray(pts);
        polyList.add(scaledPoints);
        Imgproc.drawContours(originalImage, polyList, 0, color, 2);

    }


    private MatOfPoint hullToMapOfPoint( MatOfPoint src, MatOfInt hull ){
        MatOfPoint result = new MatOfPoint();
        Point[] srcPoints = src.toArray();
        int[] hullIndicies = hull.toArray();
        Point[] resultpoints = new Point[hullIndicies.length];
        for( int i = 0; i < hullIndicies.length; i++ ){
            resultpoints[i] = srcPoints[hullIndicies[i]];
        }
        result.fromArray(resultpoints);

        return result;
    }

      /* Generate 32*32 temperature 2D array from thermal images
        Note - this is temporary as eventually the thermal map will be the input rather than an image
     */
     public int[][] getTemperatures( Mat srcImage){
        int[][] thermalMap = new int[MaxThermalRows][MaxThermalCols];
        // Note that the thermal pixels in srcImage are a 448 * 448 rectangle, centered in a 640 * 480 image
        int xyDelta = 14;   // Pixel blocks are 448/32 apart
        int rowOffset =  (480-448)/2;
        int colOffset = (640 -448)/2;
        for( int rowIndex = 0; rowIndex < MaxThermalRows; rowIndex++ ) {
            for( int colIndex = 0; colIndex < MaxThermalCols; colIndex++ ) {
                double[] RGBCenter = srcImage.get( ((rowIndex * xyDelta) + xyDelta/2)+rowOffset,
                        ((colIndex * xyDelta) + xyDelta/2) + colOffset);
                long RGBLong = (long)RGBCenter[0] << 16;
                RGBLong |= (long)RGBCenter[1] << 8;
                RGBLong |= (long)RGBCenter[2];
                try {
                    float tempFloat = PixelToTemperature.getTemperatureFromRGB(RGBLong);
                    // This is degree C. Convert to int *10
                    int tempInt = (int) (tempFloat * 10);
                    thermalMap[rowIndex][colIndex] = tempInt;
                }catch (Exception ex){
                    if (DEBUG) Log.d(TAG, ex.getMessage());
                }
            }

        }

        return thermalMap;
     }
}
