package com.example.fredoleary.thermalimage1;

import android.graphics.Bitmap;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.List;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

public class MainActivity extends AppCompatActivity {

    private static boolean DISPLAY_CONTOURS =  FALSE;            // Displays the contour lines in red
    private static boolean DISPLAY_CONTOUR_RECTS =  TRUE;      // Displays the contour covering rectangle in blue
    private static boolean DISPLAY_COVER_RECT =  FALSE;         // Displays the union of all contour covering rectangles in blue
    private static boolean DISPLAY_APPROX_CONTOURS = TRUE;     // Displays the approx lines in
    private static boolean DISPLAY_HULL = TRUE;                 // Displays the Hull in blue

    class imageEntry {
        public imageEntry(Integer resId, boolean result){
            this.resId = resId;
            this.result = result;
        }
        private Integer resId;
        private boolean result;
    }
    private static final String    TAG = "ThermalImage1";
    private Mat monoChromeImage;
    private Mat contourImage;
    private Mat originalImage;
    private FaceDetectUtil thermalUtil;
    private int nextIndex = 0;
//    private List<Integer> imageIdsOne = Arrays.asList(R.drawable.bb087);
    private boolean showAll = false;

    private List<imageEntry> oneImage = Arrays.asList(
            new imageEntry( R.drawable.s2_009, true )
    );

    /*
    Mock data for 32*32 values degree C * 10
     */
    private int[][] thermalTest =   {{ 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,000,000,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,000,000,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
            { 350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350,350},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200},
                                     { 200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200}};

    private List<imageEntry> imageEntries = Arrays.asList(
            new imageEntry( R.drawable.long_finger, false ),
            new imageEntry( R.drawable.x001, true ),
            new imageEntry( R.drawable.x002, true ),
            new imageEntry( R.drawable.x003, true ),
            new imageEntry( R.drawable.x004, true ),
            new imageEntry( R.drawable.x005, true ),
            new imageEntry( R.drawable.x006, true ),
            new imageEntry( R.drawable.x007, true ),
            new imageEntry( R.drawable.x008, true ),
            new imageEntry( R.drawable.x009, true ),
            new imageEntry( R.drawable.x010, false ),
            new imageEntry( R.drawable.x011, false ),
            new imageEntry( R.drawable.x012, false ),
            new imageEntry( R.drawable.x013, false ),
            new imageEntry( R.drawable.x014, true ),
            new imageEntry( R.drawable.x015, true ),
            new imageEntry( R.drawable.x016, true ),

            new imageEntry( R.drawable.y001, false ),
            new imageEntry( R.drawable.y007, false ),
            new imageEntry( R.drawable.z003, false ),

            new imageEntry( R.drawable.s10_011, false ),
            new imageEntry( R.drawable.s10_026, false ),
            new imageEntry( R.drawable.s10_031, false ),

            new imageEntry( R.drawable.s11_087, false ),
            new imageEntry( R.drawable.s11_096, false ),
            new imageEntry( R.drawable.s11_101, false ),
            new imageEntry( R.drawable.s11_114, false ),

            new imageEntry( R.drawable.s12_001, false ),
            new imageEntry( R.drawable.s12_003, false ),
            new imageEntry( R.drawable.s12_005, false ),

            new imageEntry( R.drawable.s1_003, false ),
//            new imageEntry( R.drawable.s1_008, false ),
            new imageEntry( R.drawable.s2_009, false ),
            new imageEntry( R.drawable.s3_004, false ),
            new imageEntry( R.drawable.s4_053, false ),
            new imageEntry( R.drawable.s4_018, false ),      // !!!!!!! should fail. Its a finger - not a head
            new imageEntry( R.drawable.s5_021, false ),
            new imageEntry( R.drawable.s5_001, false ),
            new imageEntry( R.drawable.s6_014, true ),
            new imageEntry( R.drawable.s6_061, true ),
            new imageEntry( R.drawable.s7_042, false ),
            new imageEntry( R.drawable.s13_011, false ),
            new imageEntry( R.drawable.s14_019, false ),
            new imageEntry( R.drawable.s15_001, true ),
            new imageEntry( R.drawable.s15_021, true ),
            new imageEntry( R.drawable.s15_047, true )
            );

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    nextIndex = 0;
                    processNextImage();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    private void processNextImage(){
        List<imageEntry> imgList = (showAll) ? imageEntries : oneImage;
        if( nextIndex < imgList.size()){
            final imageEntry img = imgList.get(nextIndex);
            Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                public void run() {
                    // Actions to do after 10 seconds
                    processImageEx(img);
                }
            }, 1000);
        }
    }
    private void processImageEx( imageEntry img ){
        int indexFrom1 = nextIndex +1;
        Log.d(TAG, "Processing image #" + indexFrom1);
        processImage( img );
        nextIndex++;
        processNextImage();
    }
    private void processImage( imageEntry img){
        thermalUtil = new FaceDetectUtil();
        originalImage = thermalUtil.getImage( this, img.resId);
        int[][] thermalMap = thermalUtil.getTemperatures(originalImage);
        if(originalImage != null){
            Mat[] maskAndContours = thermalUtil.getMonoChromeImageEx( originalImage, thermalMap );
            monoChromeImage = maskAndContours[0];
            contourImage = maskAndContours[1];
//            Mat monoImageInv = new Mat();
//            Core.bitwise_not ( monoChromeImage, monoImageInv );
            List<FaceDetectUtil.DetectResult> results = thermalUtil.processContours( contourImage, originalImage);
            // Draw results on originalImage
            drawResults( thermalUtil, originalImage, results );
            thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), originalImage);
            TextView resultTextView = (TextView)findViewById(R.id.imageDetect);
            int displayIndex = nextIndex + 1;
            boolean faceDetected = false;
            for( FaceDetectUtil.DetectResult result :results ){
                if( result.isFace){
                    faceDetected = true;
                    break;
                }
            }
            if( faceDetected ){
                if( img.result ) {
                    resultTextView.setText("Image detected - PASS (" + displayIndex+ ")");
                    Log.d(TAG, "Processing image #" + displayIndex + " PASSED");
                }else{
                    resultTextView.setText("Image detected - FAIL (" + displayIndex + ")");
                    Log.d(TAG, "Processing image #" + displayIndex + " FAILED");
                }
            }else{
                if( img.result ) {
                    resultTextView.setText("No Image detected - FAIL (" + displayIndex + ")");
                    Log.d(TAG, "Processing image #" + displayIndex + " FAILED");
                }else{
                    resultTextView.setText("No Image detected - PASS (" + displayIndex+ ")");
                    Log.d(TAG, "Processing image #" + displayIndex + " PASSED");
                }
            }
        }else{
            Log.e(TAG, "IMAGE NOT FOUND");

        }


    }
    /*
        Draw various image detection elements. (Note - covering rect is drawn only if a face is detected
    */

    private void drawResults(FaceDetectUtil util, Mat originalImage, List<FaceDetectUtil.DetectResult> results){
        for(FaceDetectUtil.DetectResult result : results){
            if( DISPLAY_APPROX_CONTOURS  && result.approxContour != null) {
                util.displayPoly( originalImage, result.approxContour, new Scalar(30, 255, 255) );     // white
            }
            if (DISPLAY_CONTOUR_RECTS && result.isFace && result.rect != null) {
                MatOfPoint rect = new MatOfPoint();
                Point[] rectPts = new Point[5];
                rectPts[0] = new Point( result.rect.x, result.rect.y);
                rectPts[1] = new Point( result.rect.x, result.rect.y + result.rect.height);
                rectPts[2] = new Point( result.rect.x + result.rect.width, result.rect.y + result.rect.height);
                rectPts[3] = new Point( result.rect.x + result.rect.width, result.rect.y);
                rectPts[4] = new Point( result.rect.x, result.rect.y);


                rect.fromArray( rectPts);
                util.displayPoly( originalImage, rect, new Scalar(0, 0, 255) );
            }

        }

    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

}
