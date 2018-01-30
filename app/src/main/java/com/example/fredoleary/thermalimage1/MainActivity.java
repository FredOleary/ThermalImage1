package com.example.fredoleary.thermalimage1;

import android.content.Context;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String    TAG = "ThermalImage1";
    private Mat monoChromeImage;
    private Mat contourImage;
    private Mat originalImage;
    private ThermalUtil thermalUtil;
    private int nextIndex = 0;
    private List<Integer> imageIdsOne = Arrays.asList(R.drawable.x012);
    private boolean showAll = false;

    private List<Integer> imageIds = Arrays.asList(
            R.drawable.x001, R.drawable.x002, R.drawable.x003,
            R.drawable.x004, R.drawable.x005, R.drawable.x006,
            R.drawable.x007, R.drawable.x008, R.drawable.x009,
            R.drawable.x010, R.drawable.x011, R.drawable.x012,
            R.drawable.x013, R.drawable.x014, R.drawable.x015,
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
        List<Integer> imgList = (showAll) ? imageIds : imageIdsOne;
        if( nextIndex < imgList.size()){
            final Integer resId = imgList.get(nextIndex);
            Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                public void run() {
                    // Actions to do after 10 seconds
                    processImageEx( resId);
                }
            }, 1000);
        }
    }
    private void processImageEx( Integer resId ){
        processImage( resId );
        nextIndex++;
        processNextImage();
    }
    private void processImage( Integer imageId){
        boolean detected = false;
        thermalUtil = new ThermalUtil();
        originalImage = thermalUtil.getImage( this, imageId);
        if(originalImage != null){
            Mat[] maskAndContours = thermalUtil.getMonoChromeImage( originalImage );
            monoChromeImage = maskAndContours[0];
            contourImage = maskAndContours[1];
//            detected = thermalUtil.processBlob( this, monoChromeImage, originalImage);
//            Mat monoImageInv = new Mat();
//            Core.bitwise_not ( monoChromeImage, monoImageInv );
            detected = thermalUtil.processContours( contourImage, originalImage);
            thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), originalImage);
            Log.d(TAG, "Found image");
            TextView resultTextView = (TextView)findViewById(R.id.imageDetect);
            if( detected ){
                resultTextView.setText("Image detected");
            }else{
                resultTextView.setText("No Image detected");
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
