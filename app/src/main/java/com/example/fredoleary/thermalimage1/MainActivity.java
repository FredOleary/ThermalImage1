package com.example.fredoleary.thermalimage1;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity {

    private static final String    TAG = "ThermalImage1";
    private Mat monoChromeImage;
    private Mat originalImage;
    private ThermalUtil thermalUtil;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    processImages();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    private void processImages(){
        thermalUtil = new ThermalUtil();
        originalImage = thermalUtil.getImage( this, R.drawable.x004);
        if(originalImage != null){
            monoChromeImage = thermalUtil.getMonoChromeImage( originalImage );
            thermalUtil.processContours( monoChromeImage, originalImage);
            thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), originalImage);
            Log.d(TAG, "Found image");
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
