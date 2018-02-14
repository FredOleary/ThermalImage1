package com.example.fredoleary.thermalimage1;

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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.List;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

public class MainActivity extends AppCompatActivity {

    private static boolean DISPLAY_CONTOURS =  TRUE;            // Displays the contour lines in red
    private static boolean DISPLAY_CONTOUR_RECTS =  TRUE;      // Displays the contour covering rectangle in blue
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
    private boolean showOriginalImage = false;

    private List<imageEntry> oneImage = Arrays.asList(
            new imageEntry( R.drawable.s15_047, true )
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


    private int[][] thermal_data2 =
            {{110,111,145,138,150,171,157,183,175,177,192,225,239,243,243,254,275,259,233,196,179,177,176,167,158,148,159,145,126,137,131,113},
        {125,118,153,132,163,168,180,177,197,207,230,242,248,266,277,293,291,279,253,256,220,201,176,181,191,156,155,166,136,135,129,136},
        {128,108,142,165,153,174,162,208,212,220,245,261,284,309,312,318,302,311,317,306,257,229,207,178,181,164,168,154,141,153,130,138},
        {135,131,156,149,151,163,185,200,235,263,278,294,310,330,332,316,310,328,326,324,312,259,223,194,169,173,172,158,164,147,142,130},
        {152,129,120,155,168,204,199,226,253,277,293,299,320,325,338,322,307,310,330,317,322,315,269,215,175,177,172,143,139,153,147,137},
        {149,143,136,156,186,184,198,252,295,310,318,304,307,315,316,314,309,297,324,329,328,336,309,268,220,179,176,172,163,147,139,148},
        {150,159,144,156,169,195,202,275,316,329,322,321,318,334,312,290,299,286,289,321,323,303,316,300,250,212,187,172,146,175,174,150},
        {177,153,163,159,191,197,230,286,324,316,327,322,326,317,297,287,291,286,283,302,333,322,322,315,295,213,186,159,157,171,183,167},
        {157,173,162,165,177,203,232,320,315,323,314,314,324,328,318,308,293,305,303,293,327,310,332,320,306,230,178,168,172,178,173,177},
        {158,173,175,171,177,209,282,317,312,329,323,327,327,321,318,326,328,311,315,318,323,317,312,315,310,254,184,178,171,163,175,172},
        {166,176,168,168,185,211,269,306,327,318,323,333,336,340,323,321,340,335,321,320,324,327,336,311,308,242,190,190,175,170,190,171},
        {191,192,177,185,187,213,300,319,331,312,328,330,327,335,330,335,343,342,324,334,329,317,320,322,316,249,196,169,182,183,181,177},
        {170,179,193,172,195,223,301,319,321,325,329,333,350,352,348,342,343,343,339,339,327,325,336,317,291,228,184,185,172,190,175,165},
        {175,183,184,175,201,270,312,311,306,325,327,342,342,337,331,337,346,350,331,339,333,319,322,323,308,222,194,168,187,176,172,164},
        {175,182,192,212,220,296,305,321,333,323,351,351,339,345,332,345,337,338,340,336,342,334,321,307,289,210,185,189,175,199,194,184},
        {179,164,172,184,200,267,287,297,307,312,321,327,337,327,326,330,332,330,326,336,328,337,327,321,278,209,199,184,188,171,189,167},
        {167,182,174,189,191,242,271,308,317,315,328,333,330,321,338,310,323,321,321,328,332,314,324,309,271,202,195,181,192,194,178,174},
        {168,168,170,191,180,190,238,297,317,320,323,329,332,329,327,329,323,316,317,319,329,333,324,318,252,192,189,182,190,184,169,165},
        {174,176,174,183,188,188,213,301,307,333,322,320,323,329,336,334,327,327,329,327,321,323,318,310,238,193,186,188,195,185,188,169},
        {175,177,183,181,184,197,204,313,315,321,321,328,331,340,335,331,330,324,330,322,318,321,303,272,219,182,194,176,190,174,166,181},
        {190,164,175,174,170,182,195,295,327,310,315,328,331,338,336,352,342,331,341,329,321,320,314,243,191,182,185,191,190,193,183,179},
        {182,162,172,172,164,185,210,289,317,308,323,329,333,342,338,341,346,333,327,319,325,331,277,194,182,176,195,186,185,170,187,175},
        {152,173,160,166,163,192,216,298,312,327,311,333,314,336,343,338,335,329,326,325,324,322,244,196,179,190,182,195,181,173,183,157},
        {173,168,179,197,217,230,289,311,307,324,319,317,323,324,334,325,328,335,323,301,307,295,221,192,181,188,197,186,166,178,176,173},
        {179,190,213,219,261,286,321,334,328,326,317,325,319,342,324,316,283,273,233,233,219,219,229,203,207,193,199,185,198,179,193,179},
        {203,210,219,209,236,278,296,327,328,323,325,314,323,326,296,253,247,244,232,213,196,205,192,196,202,201,193,194,172,173,165,177},
        {222,212,206,217,221,225,244,249,280,317,293,283,322,287,267,232,220,213,196,197,201,192,191,198,198,194,199,204,193,191,180,158},
        {219,208,226,218,222,221,226,239,240,272,259,226,211,219,215,203,196,185,196,182,192,188,203,205,225,222,218,216,236,200,199,173},
        {205,228,229,246,226,235,228,229,231,208,221,199,208,207,211,214,220,195,194,219,221,208,216,219,221,219,212,221,233,242,211,184},
        {219,219,208,215,210,242,221,227,232,221,202,217,220,224,225,217,218,214,226,226,219,215,207,216,240,255,231,229,217,224,224,213},
        {210,213,228,236,209,221,235,233,234,226,215,213,219,220,217,227,205,204,202,224,221,248,225,228,255,265,235,217,203,210,218,194},
        {207,208,211,225,221,235,246,227,237,206,218,198,209,203,208,215,217,221,241,224,224,231,215,218,226,248,243,221,196,185,206,192}};

    private int[][] thermal_data3 =
            {{109,127,141,123,156,178,165,187,183,184,191,220,232,244,256,251,267,257,238,203,187,201,177,154,158,161,160,146,124,129,137,117},
        {129,121,152,143,168,161,168,181,195,206,235,248,237,270,284,297,270,275,252,255,224,184,163,191,186,156,157,141,154,147,123,150},
        {145,106,138,165,168,181,152,207,205,222,249,263,282,308,308,321,312,316,316,288,251,221,206,169,177,179,158,170,148,167,127,133},
        {137,147,161,147,166,163,196,211,230,260,265,285,314,328,319,322,305,325,319,318,313,263,220,191,158,164,178,156,152,150,134,126},
        {148,141,142,155,155,199,204,211,260,276,292,308,310,331,343,320,310,311,317,319,321,320,262,230,175,182,174,151,151,139,145,163},
        {157,141,137,141,184,178,199,253,296,305,311,310,304,313,316,303,298,302,327,327,324,328,297,268,223,176,180,161,170,132,146,146},
        {148,153,164,148,181,204,207,289,316,324,315,317,313,329,326,304,299,281,292,324,324,309,317,299,259,212,192,169,157,179,180,144},
        {162,144,163,153,187,206,247,295,332,317,326,325,330,323,299,295,295,287,282,299,338,320,330,317,294,216,190,169,168,172,176,168},
        {157,164,159,169,191,202,253,315,306,321,321,307,321,332,327,314,300,314,308,295,327,312,328,304,311,237,186,182,172,171,151,169},
        {163,186,183,183,182,213,274,317,310,323,309,333,330,322,326,327,319,318,322,312,317,316,314,305,312,247,188,190,168,168,161,171},
        {181,179,171,180,197,218,276,314,326,319,320,323,336,335,325,331,341,330,318,314,322,328,327,307,314,243,191,186,183,176,179,169},
        {185,169,171,183,184,220,303,319,324,323,328,325,332,338,332,328,337,346,322,321,323,318,322,313,320,237,180,185,180,166,175,174},
        {186,189,181,183,194,230,315,316,316,327,332,334,345,347,356,343,340,349,343,330,329,326,330,319,293,237,189,177,171,177,186,169},
        {193,186,193,193,204,276,312,316,311,321,334,343,337,343,334,344,349,340,339,346,335,326,327,326,298,219,202,171,171,180,165,178},
        {179,188,195,200,232,297,303,324,331,324,354,339,344,342,333,349,337,349,338,336,349,337,329,320,284,214,200,197,189,191,187,192},
        {166,167,179,185,201,261,290,296,303,310,316,341,328,329,320,329,335,335,318,340,334,330,312,314,274,212,203,187,191,179,179,164},
        {172,175,174,173,194,245,261,312,309,327,327,347,323,322,337,310,332,314,314,332,333,314,325,303,271,202,185,179,208,181,171,186},
        {179,156,181,186,193,187,238,299,313,328,330,333,333,328,322,328,313,315,322,317,327,321,326,310,248,183,190,179,201,191,174,163},
        {177,173,179,187,181,190,208,297,308,335,321,316,324,321,330,325,328,331,329,334,314,320,309,295,241,189,191,193,196,183,195,158},
        {164,188,179,171,181,184,208,321,311,327,322,338,337,345,341,345,340,329,334,318,320,328,306,264,213,188,197,175,180,179,164,178},
        {190,154,166,171,179,191,204,305,316,318,314,333,332,337,338,357,352,327,352,334,320,313,301,234,182,190,187,177,179,179,175,179},
        {180,154,167,177,170,196,221,296,329,316,324,318,337,333,333,335,342,328,324,325,330,326,281,191,176,183,186,172,184,166,188,177},
        {147,191,169,166,175,196,215,309,314,317,312,334,328,335,342,331,339,331,324,324,333,325,240,185,187,177,191,202,177,190,183,138},
        {179,161,184,191,210,238,291,311,317,312,327,318,325,317,344,331,328,332,325,311,297,284,205,199,190,184,206,186,175,167,189,178},
        {188,199,197,214,258,296,326,334,330,335,317,324,329,336,325,324,295,275,236,238,229,232,225,198,203,198,201,180,185,159,161,169},
        {212,215,217,209,238,272,298,326,318,326,325,320,324,316,298,255,243,247,233,217,216,201,183,200,183,195,194,185,172,186,170,187},
        {215,204,221,222,229,230,243,261,293,331,296,287,318,288,253,235,235,219,205,189,205,198,193,181,191,195,191,201,205,177,165,161},
        {216,212,226,218,234,214,229,237,249,265,259,237,213,218,212,203,198,189,202,190,201,187,198,211,224,225,214,216,236,198,191,189},
        {209,235,220,238,226,228,241,221,234,206,218,210,204,221,208,216,209,202,190,217,221,196,199,207,214,227,206,219,249,225,202,187},
        {215,232,211,222,214,246,225,231,229,221,202,226,218,221,215,204,220,216,223,213,219,211,216,205,235,254,226,232,228,242,209,218},
        {220,231,223,241,210,233,226,239,225,234,218,216,224,216,231,219,202,211,207,234,226,248,209,232,257,252,223,204,206,223,202,196},
        {217,210,219,223,219,232,226,226,225,210,219,216,196,199,210,214,219,215,247,233,233,226,225,221,226,247,235,218,212,181,199,201}};

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
            new imageEntry( R.drawable.s11_087, false ),
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
//            Mat[] maskAndContours = thermalUtil.getMonoChromeImageEx( thermalMap );
            Mat[] maskAndContours = thermalUtil.getMonoChromeImageEx( thermal_data3 );
            monoChromeImage = maskAndContours[0];
            contourImage = maskAndContours[1];

            List<FaceDetectUtil.DetectResult> results = thermalUtil.processContours( contourImage );
            if( showOriginalImage) {
                // Draw results on originalImage
                drawResults(thermalUtil, originalImage, results);
                thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), originalImage);
                TextView resultTextView = findViewById(R.id.imageDetect);
                int displayIndex = nextIndex + 1;
                boolean faceDetected = false;
                for (FaceDetectUtil.DetectResult result : results) {
                    if (result.isFace) {
                        faceDetected = true;
                        break;
                    }
                }
                if (faceDetected) {
                    if (img.result) {
                        resultTextView.setText("Image detected - PASS (" + displayIndex + ")");
                        Log.d(TAG, "Processing image #" + displayIndex + " PASSED");
                    } else {
                        resultTextView.setText("Image detected - FAIL (" + displayIndex + ")");
                        Log.d(TAG, "Processing image #" + displayIndex + " FAILED");
                    }
                } else {
                    if (img.result) {
                        resultTextView.setText("No Image detected - FAIL (" + displayIndex + ")");
                        Log.d(TAG, "Processing image #" + displayIndex + " FAILED");
                    } else {
                        resultTextView.setText("No Image detected - PASS (" + displayIndex + ")");
                        Log.d(TAG, "Processing image #" + displayIndex + " PASSED");
                    }
                }
            }else{
                // For debug. Show the 32*32 image instead of the original
                debugMonoImage(results, contourImage);
            }
        }else{
            Log.e(TAG, "IMAGE NOT FOUND");

        }

    }
    private void debugMonoImage( List<FaceDetectUtil.DetectResult> results, Mat debugImage){
        for(FaceDetectUtil.DetectResult result : results){
            Rect rect = Imgproc.boundingRect( result.contour);
            Imgproc.rectangle(
                    debugImage,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 255, 255),
                    1);



        }
        thermalUtil.displayImage(this, (ImageView) (this.findViewById(R.id.imgview)), debugImage);
    }
    /*
        Draw various image detection elements. (Note - covering rect is drawn only if a face is detected
    */

    private void drawResults(FaceDetectUtil util, Mat originalImage, List<FaceDetectUtil.DetectResult> results){
        for(FaceDetectUtil.DetectResult result : results){
            if( DISPLAY_CONTOURS && result.contour != null ){
                util.displayPoly( originalImage, result.contour, new Scalar(255, 0, 0) );     // red

            }
            if( DISPLAY_APPROX_CONTOURS  && result.approxContour != null) {
                util.displayPoly( originalImage, result.approxContour, new Scalar(30, 255, 255) );     // acqa
            }

            if( DISPLAY_HULL && result.hull != null ){
                util.displayPoly( originalImage, result.hull, new Scalar(0, 0, 255) );     // blue
            }
            if (DISPLAY_CONTOUR_RECTS && result.rect != null) {
                MatOfPoint rect = new MatOfPoint();
                Point[] rectPts = new Point[5];
                rectPts[0] = new Point( result.rect.x, result.rect.y);
                rectPts[1] = new Point( result.rect.x, result.rect.y + result.rect.height);
                rectPts[2] = new Point( result.rect.x + result.rect.width, result.rect.y + result.rect.height);
                rectPts[3] = new Point( result.rect.x + result.rect.width, result.rect.y);
                rectPts[4] = new Point( result.rect.x, result.rect.y);


                rect.fromArray( rectPts);
                util.displayPoly( originalImage, rect, new Scalar(0, 0, 255) ); // Blue
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
