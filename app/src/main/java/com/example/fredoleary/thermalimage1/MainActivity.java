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

import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
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
    private boolean showAll = true;

    private List<imageEntry> oneImage = Arrays.asList(
            new imageEntry( R.drawable.x014, true )
    );

    private List<imageEntry> imageEntries = Arrays.asList(

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
            new imageEntry( R.drawable.x011, true ),
            new imageEntry( R.drawable.x012, false ),
            new imageEntry( R.drawable.x013, false ),
            new imageEntry( R.drawable.x014, true ),
            new imageEntry( R.drawable.x015, false ),
            new imageEntry( R.drawable.x016, true ),

            new imageEntry( R.drawable.y001, false ),
            new imageEntry( R.drawable.y007, false ),
            new imageEntry( R.drawable.z003, false ),

            new imageEntry( R.drawable.aa011, false ),
            new imageEntry( R.drawable.aa026, false ),
            new imageEntry( R.drawable.aa031, false ),

            new imageEntry( R.drawable.bb087, false ),
            new imageEntry( R.drawable.bb096, false ),
            new imageEntry( R.drawable.bb101, false ),

            new imageEntry( R.drawable.cc001, false ),
            new imageEntry( R.drawable.cc003, false ),
            new imageEntry( R.drawable.cc005, false )

            );
    private List<Integer> imageIds = Arrays.asList(
            R.drawable.x001, R.drawable.x002, R.drawable.x003,
            R.drawable.x004, R.drawable.x005, R.drawable.x006,
            R.drawable.x007, R.drawable.x008, R.drawable.x009,
            R.drawable.x010, R.drawable.x011, R.drawable.x012,
            R.drawable.x013, R.drawable.x014, R.drawable.x015,
            R.drawable.y007, R.drawable.y001, R.drawable.z003,
            R.drawable.aa011, R.drawable.aa026, R.drawable.aa031,
            R.drawable.bb096, R.drawable.bb101, R.drawable.bb087,
            R.drawable.cc001, R.drawable.cc003, R.drawable.cc005,
            R.drawable.x016, R.drawable.testx001 );

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
        processImage( img );
        nextIndex++;
        processNextImage();
    }
    private void processImage( imageEntry img){
        Bitmap imageDetected = null;
        thermalUtil = new FaceDetectUtil();
        originalImage = thermalUtil.getImage( this, img.resId);
        if(originalImage != null){
            Mat[] maskAndContours = thermalUtil.getMonoChromeImage( originalImage );
            monoChromeImage = maskAndContours[0];
            contourImage = maskAndContours[1];
//            detected = thermalUtil.processBlob( this, monoChromeImage, originalImage);
//            Mat monoImageInv = new Mat();
//            Core.bitwise_not ( monoChromeImage, monoImageInv );
            imageDetected = thermalUtil.processContours( contourImage, originalImage);
            thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), originalImage);
            TextView resultTextView = (TextView)findViewById(R.id.imageDetect);
            int displayIndex = nextIndex + 1;
            if( imageDetected != null ){
                if( img.result ) {
                    resultTextView.setText("Image detected - PASS (" + displayIndex+ ")");
                }else{
                    resultTextView.setText("Image detected - FAIL (" + displayIndex + ")");
                }
            }else{
                if( img.result ) {
                    resultTextView.setText("No Image detected - FAIL (" + displayIndex + ")");
                }else{
                    resultTextView.setText("No Image detected - PASS (" + displayIndex+ ")");
                }
            }
        }else{
            Log.e(TAG, "IMAGE NOT FOUND");

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
