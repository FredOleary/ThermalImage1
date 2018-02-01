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
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static org.opencv.features2d.FeatureDetector.FAST;
import static org.opencv.features2d.Features2d.DRAW_OVER_OUTIMG;
import static org.opencv.imgproc.Imgproc.MORPH_DILATE;


/**
 * Created by fredoleary on 1/24/18.
 */

public class FaceDetectUtil {
    /*
    Flags for various processing
     */
    private static boolean DISPLAY_CONTOURS =  FALSE;            // Displays the contour lines in red
    private static boolean DISPLAY_CONTOUR_RECTS =  TRUE;      // Displays the contour covering rectangle in blue
    private static boolean DISPLAY_COVER_RECT =  FALSE;         // Displays the union of all contour covering rectangles in blue
    private static boolean DISPLAY_APPROX_CONTOURS = TRUE;     // Displays the approx lines in
    private static boolean DISPLAY_HULL = TRUE;                 // Displays the Hull in blue

    private static boolean DISPLAY_EXTRA_LOGS = true;
    private int[] RGBMinRed = new int[3];
    private int[] RGBMaxRed = new int[3];

    private int[] RGBMinGreen = new int[3];
    private int[] RGBMaxGreen = new int[3];

    private int[] RGBMinBlue = new int[3];
    private int[] RGBMaxBlue = new int[3];

    private static final String    TAG = FaceDetectUtil.class.getName();

    /* HSV filter colors. Note these are derived empiracally from paint brush.
     RANGES:
        Paintbrush uses 0-360 range for Hue (H), openCV uses 0-180 range
        Paintbrush uses 0-100 range for Saturation/Value (H/V), openCV uses 0-255 range

        From Temperature to Color convertor
        rgb=E0CE00 r=224 g=206 b=0 temperature=32.0, Min temp
        rgb=E06600 r=224 g=102 b=0 temperature=40.0, Max temp

        rgb=E0CE00 in HSV H = 55 (In range 0-360), S = 100%, V = 88%
        rgb=E06600 in HSV H = 27 (In range 0-360), S = 100%, V = 88%

        Note: Empirically a range of S/V values is required, E.g. 90-100 %
     */
    private Scalar low_color = new Scalar(27.0/360*180, 90.0/100*255, 87.0/100*255);
    private  Scalar high_color = new Scalar(55.0/360*180, 100.0/100*255, 89.0/100*255);

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
    private static final int minWidthPct =30;

    /*
    Image width cannoy exceed maxScreenPct of the screen width
     */
    private static final int maxScreenPct = 90;

    public Bitmap detectFaces(Bitmap bitmap) {
        Mat originalImage = new Mat();
        Utils.bitmapToMat(bitmap, originalImage);
        Mat[] maskAndContours = getMonoChromeImage(originalImage);
//        Mat monoChromeImage = maskAndContours[0];
        Mat contourImage = maskAndContours[1];
//            detected = processBlob( this, monoChromeImage, originalImage);
//            Mat monoImageInv = new Mat();
//            Core.bitwise_not ( monoChromeImage, monoImageInv );
        Bitmap detectedImage = processContours(contourImage, originalImage);
        Log.d(TAG, "detectFaces detected=" + (detectedImage != null ? true : false));
        return detectedImage;
    }

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

    /*
    Mat[0] = Mask (White pixels are image of interest
    Mat[1] = Contours
     */
    public Mat[] getMonoChromeImage(Mat image ){
        Mat hsvImage;
        hsvImage = new Mat();
        Mat  blurredImage = new Mat();
        Mat[] results= new Mat[2];

        //Imgproc.blur(image, blurredImage, new Size(100, 100));
        // Convert to HSV for filtering

        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_RGB2HSV);
        Mat mask = new Mat();

        Core.inRange(hsvImage, low_color, high_color, mask);

        int morph_size = 5;
        Mat element = Imgproc.getStructuringElement( Imgproc.MORPH_RECT, new Size( 2*morph_size + 1, 2*morph_size+1 ), new Point( morph_size, morph_size ) );

        Mat morphedMask = new Mat(); // result matrix

        Point kernel = new Point(-1,-1);
        Imgproc.morphologyEx( mask, morphedMask, MORPH_DILATE, element, kernel, 3);

        results[0] = morphedMask;

        // Use Canny to create the contours on the monochrome image
        Mat contourImage = new Mat();
        Imgproc.Canny(morphedMask, contourImage, 0, 40);
        results[1] = contourImage;
        return results;

    }

    public boolean processBlob( Context context, Mat monoImage, Mat originalImage ) {
        MatOfKeyPoint matOfKeyPoints = new MatOfKeyPoint();

        FeatureDetector blobDetector = FeatureDetector.create(FAST);
        File file = context.getFilesDir();
        String path = file.getPath();
        Mat monoImageInv = new Mat();
        try {

            path += "/Fred_test";
            blobDetector.write(path);

            String config = getFileContents(new File(path));
            blobDetector.read(path);

            Core.bitwise_not ( monoImage, monoImageInv );

            blobDetector.detect(monoImage, matOfKeyPoints);

            Features2d.drawKeypoints(monoImage, matOfKeyPoints, originalImage, new Scalar(0, 0, 255), DRAW_OVER_OUTIMG);
            Log.d(TAG, "foo");
        }catch (IOException ex ){
            ex.printStackTrace();
        }
        return false;
    }

    public Bitmap processContours( Mat monoImage, Mat originalImage ) {
        Log.d(TAG, "processContours - Begin ------------------------------" );
        Bitmap retResult = null;

        if( DISPLAY_EXTRA_LOGS){
            showExtraLogs(originalImage );
            showRGBColorRange( originalImage);
        }

        List<MatOfPoint> contours = new ArrayList();
        Mat hierarchy = new Mat();
        Imgproc.findContours(monoImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Log.d( TAG, "findContours returned " +  contours.size() + " contours");
        Mat detectedImage = null;
        for (MatOfPoint contour : contours) {
            Mat result = processContour(contour, monoImage, originalImage);
            if(result != null ){
                detectedImage = result;
            }
        }
        if (detectedImage != null) {
            Bitmap bitMap = Bitmap.createBitmap(detectedImage.cols(),
                    detectedImage.rows(),Bitmap.Config.RGB_565);

            Utils.matToBitmap(detectedImage, bitMap);
            retResult = bitMap;
        }
        Log.d(TAG, "processContours - End ------------------------------" );
        return retResult;
    }

    private Mat processContour( MatOfPoint contour, Mat monoImage, Mat originalImage ){
        boolean faceDetected = false;

        Rect boundingRect = Imgproc.boundingRect(contour);
        int imageSize = monoImage.width() * monoImage.height();

        // Check if the contour is large enough

        if( DISPLAY_CONTOURS ) {
            displayPoly( originalImage, monoImage, contour, new Scalar(255, 0, 0) );     // Red
        }
        int rectAreaPct = (int)(boundingRect.area()/(double)imageSize * 100);
        if( rectAreaPct >  minRectSizePct &&  rectAreaPct < maxRectSizePct) {
            Log.d(TAG, "Rectangle included. Area%: " + rectAreaPct);

            // approximates a polygonal curve with the specified precision
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(curve, approxCurve, 0.01 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();

            // Require at least 5 vertices... (empirical)
            if( numberVertices > 5){
                Log.d(TAG, "Shape included. numberVertices: " + numberVertices);

                MatOfPoint approxCurveInt = new MatOfPoint(approxCurve.toArray());
                if( DISPLAY_APPROX_CONTOURS ) {
                    displayPoly( originalImage, monoImage, approxCurveInt, new Scalar(30, 255, 255) );     // white
                }

                // Get the hull
                MatOfInt hull = new MatOfInt();
                Imgproc.convexHull(approxCurveInt, hull);

                MatOfPoint hullPoints = hullToMapOfPoint(approxCurveInt, hull );
                 if( DISPLAY_HULL ) {
                    displayPoly( originalImage, monoImage, hullPoints, new Scalar(0, 0, 255) );     // Blue
                }

                double hullArea = Imgproc.contourArea( hullPoints);
                double polyArea = Imgproc.contourArea( approxCurveInt);
                int areaRatioPct = (int)(polyArea/hullArea *100);

//                Log.d(TAG, "hullArea: " + hullArea +
//                        ". polyArea " + polyArea +
//                        ". areaRatioPct: " + areaRatioPct);

                if( areaRatioPct > 85 ){
                    Log.d(TAG, "Shape MATCH: areaRatioPct " + areaRatioPct );

                    // Checking that width fits withing screen. (Too wide is rejected)
                    int screenWidth = monoImage.width();
                    if( screenWidth == 640 ){
                        // There is an 80 pixel, left/right margin... (TODO CLeanup)
                        screenWidth -= 2 *80;
                    }
                    int screenPct = (int)((double)boundingRect.width/(double)screenWidth*100);

                    if(screenPct < maxScreenPct ) {
                        Log.d(TAG, "Image width OK: maxScreenPct " + screenPct);

                        // Checking ratio of width to height
                        int widthPct = (int) ((double) boundingRect.width / (double) boundingRect.height * 100);
                        if (widthPct <= maxWidthPct && widthPct >= minWidthPct) {
                            Log.d(TAG, "Width/Height MATCH: widthPct " + widthPct);
                            faceDetected = true;
                            if (DISPLAY_CONTOUR_RECTS) {
                                Imgproc.rectangle(
                                        originalImage,
                                        new Point(boundingRect.x, boundingRect.y),
                                        new Point(boundingRect.x + boundingRect.width, boundingRect.y + boundingRect.height),
                                        new Scalar(0, 0, 255),
                                        2);
                            }
                        } else {
                            Log.d(TAG, "Width/Height MISMATCH: widthPct " + widthPct);
                        }
                    }else{
                        Log.d(TAG, "Image too WIDE: maxScreenPct " + screenPct);
                    }

                }else{
                    Log.d(TAG, "Shape MISMATCH: areaRatioPct " + areaRatioPct );
                }
                return faceDetected ? originalImage : null;

//                // test defects from hull
//                MatOfInt4 convexityDefects = new MatOfInt4();
//                Imgproc.convexityDefects(approxCurveInt, hull, convexityDefects);
//                faceDetected = true;
//                if( convexityDefects.total() > 0 ) {
//                    double acculatedDistance = 0;
//                    int defects[] = convexityDefects.toArray();
//                    // see https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=convexhull#convexitydefects
//                    // For details on the format, (ugly) of results of convexityDefects
//                    for (int idx = 3; idx < defects.length; idx += 4) {
//                        double distance = (double) defects[idx] / 256;
//                        Log.d(TAG, "Distance: " + distance);
//                        acculatedDistance += distance;
//                    }
//                    double defectPct = acculatedDistance/boundingRect.area() *100;
//                    Log.d(TAG, "acculatedDistance: " + acculatedDistance +
//                             ". boundingRect.width " + boundingRect.width +
//                             ". boundingRect.height " + boundingRect.height +
//                             ". defectPct: " + defectPct);
//
//                    if(defectPct > 0.9 ){       // Empirical
//                        faceDetected = false;
//                    }else{
//                        if(DISPLAY_CONTOUR_RECTS) {
//                            Imgproc.rectangle(
//                                    originalImage,
//                                    new Point(boundingRect.x, boundingRect.y),
//                                    new Point(boundingRect.x + boundingRect.width, boundingRect.y + boundingRect.height),
//                                    new Scalar(0, 0, 255),
//                                    2);
//                        }
//
//                    }
//                }

            }else{
                Log.d(TAG, "Contour excluded. Too few numberVertices: " + numberVertices);
            }

        }else{
            Log.d(TAG, "Rectangle excluded. Area too small/large, Area%: " + rectAreaPct);
        }
        return null;
    }

    private void displayPoly( Mat originalImage, Mat monoImage, MatOfPoint poly, Scalar color ){
        List<MatOfPoint> polyList = new ArrayList<>();
        polyList.add(poly);
        Imgproc.drawContours(originalImage, polyList, 0, color, 2);  // voilet

    }

    private void showExtraLogs( Mat originalImage){
        int x = originalImage.width()/2;
        int y = originalImage.height()/2;
        double[] RGBCenter = originalImage.get( x, y);
        Log.d(TAG, "RGB of center: R=" + (int)RGBCenter[0] + " G=" + (int)RGBCenter[1] + " B=" + (int)RGBCenter[2]);
        RGBCenter[0] = RGBCenter[1] = RGBCenter[2] = 0;

//        Imgproc.rectangle(
//                originalImage,
//                new Point(x, y),
//                new Point(x+2, y+2),
//                new Scalar(0, 0, 255),
//                1);

    }

    private void showRGBColorRange( Mat rgbImage ){
        // Note that thermal image is 448 * 448 centered in a 640 * 480 rectangle. Skip the frame # at top left (24 rows)
        int rowStart = 40;
        int rowEnd = (480-448)/2  + 448 - (40-16);
        boolean first = true;
        while(rowStart < rowEnd ){
            int colStart = (640-448)/2;
            while( colStart < ((640-448)/2+448) ) {
                try {
                    double[] RGBvalue = rgbImage.get(rowStart, colStart);

                    //Log.d(TAG, "RGB of at x=" + rowStart + " y=" + colStart + ": R=" + (int) RGBCenter[0] + " G=" + (int) RGBCenter[1] + " B=" + (int) RGBCenter[2]);
                    if( first){
                        first = false;
                        RGBMaxRed[0] = (int)RGBvalue[0];
                        RGBMaxRed[1] = (int)RGBvalue[1];
                        RGBMaxRed[2] = (int)RGBvalue[2];

                        RGBMinRed[0] = (int)RGBvalue[0];
                        RGBMinRed[1] = (int)RGBvalue[1];
                        RGBMinRed[2] = (int)RGBvalue[2];

                        RGBMaxGreen[0] = (int)RGBvalue[0];
                        RGBMaxGreen[1] = (int)RGBvalue[1];
                        RGBMaxGreen[2] = (int)RGBvalue[2];

                        RGBMinGreen[0] = (int)RGBvalue[0];
                        RGBMinGreen[1] = (int)RGBvalue[1];
                        RGBMinGreen[2] = (int)RGBvalue[2];

                        RGBMaxBlue[0] = (int)RGBvalue[0];
                        RGBMaxBlue[1] = (int)RGBvalue[1];
                        RGBMaxBlue[2] = (int)RGBvalue[2];

                        RGBMinBlue[0] = (int)RGBvalue[0];
                        RGBMinBlue[1] = (int)RGBvalue[1];
                        RGBMinBlue[2] = (int)RGBvalue[2];

                    }else{
                        updateMinMax( 0, RGBMinRed, RGBMaxRed, RGBvalue);
                        updateMinMax( 1, RGBMinGreen, RGBMaxGreen, RGBvalue);
                        updateMinMax( 2, RGBMinBlue, RGBMaxBlue, RGBvalue);

                    }
                    colStart++;
                }catch(Exception ex ){
                    Log.d(TAG, "foo");
                }
            }
            rowStart++;
        }
        Log.d(TAG, "--RGB of Max Red: R=" + (int) RGBMaxRed[0] + " G=" + (int) RGBMaxRed[1] + " B=" + (int) RGBMaxRed[2]);
        Log.d(TAG, "--RGB of Min Red: R=" + (int) RGBMinRed[0] + " G=" + (int) RGBMinRed[1] + " B=" + (int) RGBMinRed[2]);

        Log.d(TAG, "--RGB of Max Green: R=" + (int) RGBMaxGreen[0] + " G=" + (int) RGBMaxGreen[1] + " B=" + (int) RGBMaxGreen[2]);
        Log.d(TAG, "--RGB of Min Green: R=" + (int) RGBMinGreen[0] + " G=" + (int) RGBMinGreen[1] + " B=" + (int) RGBMinGreen[2]);

        Log.d(TAG, "--RGB of Max Blue: R=" + (int) RGBMaxBlue[0] + " G=" + (int) RGBMaxBlue[1] + " B=" + (int) RGBMaxBlue[2]);
        Log.d(TAG, "--RGB of Min Blue: R=" + (int) RGBMinBlue[0] + " G=" + (int) RGBMinBlue[1] + " B=" + (int) RGBMinBlue[2]);

    }
    private void updateMinMax( int idx, int[] min, int[] max, double[] val){
        if( (int)val[idx] < min[idx] && (int)val[0] > 40){
            min[0] = (int)val[0];
            min[1] = (int)val[1];
            min[2] = (int)val[2];
        }
        if( (int)val[idx] > max[idx]){
            max[0] = (int)val[0];
            max[1] = (int)val[1];
            max[2] = (int)val[2];
        }

    }

    private Rect unionRect( Rect r1, Rect r2 ){
        int x, y, w, h;
        x = Math.min(r1.x, r2.x);
        y = Math.min(r1.y, r2.y);
        w = Math.max(r1.x+r1.width, r2.x+r2.width) - x;
        h = Math.max(r1.y+r1.height, r2.y+r2.height) - y;
        return new Rect(x, y, w, h);

    }

     private String getFileContents(final File file) throws IOException {
        final InputStream inputStream = new FileInputStream(file);
        final BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

        final StringBuilder stringBuilder = new StringBuilder();

        boolean done = false;

        while (!done) {
            final String line = reader.readLine();
            done = (line == null);

            if (line != null) {
                stringBuilder.append(line);
            }
        }

        reader.close();
        inputStream.close();

        return stringBuilder.toString();


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

    private  void testConvexityDefects() {
        MatOfPoint points = new MatOfPoint(
                new Point(20, 0),
                new Point(40, 0),
                new Point(30, 20),
                new Point(0,  20),
                new Point(20, 10),
                new Point(30, 10)
        );


        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(points, hull);

        MatOfInt4 convexityDefects = new MatOfInt4();
        Imgproc.convexityDefects(points, hull, convexityDefects);
        Log.d(TAG, "Whoo-hoo");
     }
}
