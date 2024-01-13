package com.example.pose


import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.Button
import android.widget.Toast
import androidx.core.content.getSystemService
import com.example.pose.ml.AutoModel1
import com.example.pose.ml.AutoModel4
import kotlinx.coroutines.sync.Semaphore
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.pow

class MainActivity : AppCompatActivity() {

    val paint= Paint()

    val REQUEST_CODE = 2000

    lateinit var captureButton : Button
    lateinit var imageProcessor: ImageProcessor
    lateinit var model :AutoModel4
    lateinit var bitmap:Bitmap
    lateinit var imageView:ImageView
    lateinit var handler: Handler
    lateinit var handlerThread: HandlerThread
    lateinit var textureView: TextureView
    lateinit var cameraManager: CameraManager
    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null
    val SHOULDER_KEYPOINT_INDEX = 5
    val HIP_KEYPOINT_INDEX = 11
    val PROPORTION_THRESHOLD = 0.2
    val cameraOpenCloseLock = Semaphore(1)



    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        get_permissions()

        captureButton = findViewById(R.id.captureButton)
        imageProcessor=ImageProcessor.Builder().add(ResizeOp(192,192,ResizeOp.ResizeMethod.BILINEAR)).build()
        model = AutoModel4.newInstance(this)
        imageView=findViewById(R.id.captureImageView)
        textureView=findViewById(R.id.textureView)
        cameraManager=getSystemService(Context.CAMERA_SERVICE)as CameraManager
        handlerThread= HandlerThread("videoThread")
        handlerThread.start()
        handler=Handler(handlerThread.looper)

        paint.setColor(Color.YELLOW)

        fun calculateDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
            return Math.sqrt((x2 - x1).toDouble().pow(2) + (y2 - y1).toDouble().pow(2)).toFloat()
        }

        fun assessBodyProportions(outputFeature0: FloatArray) {
            val shoulderX = outputFeature0[SHOULDER_KEYPOINT_INDEX * 3 + 1]
            val shoulderY = outputFeature0[SHOULDER_KEYPOINT_INDEX * 3]
            val hipX = outputFeature0[HIP_KEYPOINT_INDEX * 3 + 1]
            val hipY = outputFeature0[HIP_KEYPOINT_INDEX * 3]

            val distanceBetweenShouldersAndHips = calculateDistance(shoulderX, shoulderY, hipX, hipY)

            if (distanceBetweenShouldersAndHips < PROPORTION_THRESHOLD) {
                showProportionAssessment("Good body proportions!")
            } else {
                showProportionAssessment("Body proportions need improvement.")
            }
        }

        textureView.surfaceTextureListener=object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0:SurfaceTexture,p1:Int,p2:Int)

            {
                open_camera()

            }

            override fun onSurfaceTextureSizeChanged(p0:SurfaceTexture,p1:Int,p2:Int)

            {


            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {


                bitmap=textureView.bitmap!!
                var tensorImage=TensorImage(DataType.UINT8)
                tensorImage.load(bitmap)
                tensorImage= imageProcessor.process(tensorImage)



// Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
                inputFeature0.loadBuffer(tensorImage.buffer)


                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                var mutable=bitmap.copy(Bitmap.Config.ARGB_8888,true)
                var canvas=Canvas(mutable)

                var h=bitmap.height
                var w=bitmap.width
                var x=0
                while(x<=49){
                    if(outputFeature0.get(x+2)>0.45){
                        val centerX = outputFeature0.get(x + 1) * w
                        val centerY = outputFeature0.get(x) * h
                        canvas.drawCircle(centerX, centerY, 10f, paint)
                        if (x > 2) {
                            val prevX = outputFeature0.get(x - 2) * w
                            val prevY = outputFeature0.get(x - 3) * h
                            canvas.drawLine(prevX, prevY, centerX, centerY, paint)
                        }

                    }
                    x+=3
                }
                imageView.setImageBitmap(mutable)
                assessBodyProportions(outputFeature0)


            }
        }
        captureButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, REQUEST_CODE)
        }


    }


    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }


    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0],object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                var captureRequest=p0.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                var surface=Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)

                p0.createCaptureSession(listOf(surface),object:CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(),null,null)

                    }

                    override fun onConfigureFailed(p0: CameraCaptureSession) {

                    }
                },handler)

            }


            override fun onDisconnected(camera: CameraDevice) {

            }

            override fun onError(camera: CameraDevice, error: Int) {

            }
        },handler)
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK && requestCode == REQUEST_CODE && data != null) {
            val capturedBitmap = data.extras!!.get("data") as Bitmap
            processAndAssessProportions(capturedBitmap)
        }
    }

    private fun processAndAssessProportions(capturedBitmap: Bitmap) {
        // Process the captured image with the TensorFlow Lite model
        var tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(capturedBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
        inputFeature0.loadBuffer(tensorImage.buffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        // Assess body proportions based on the distance between shoulders and hips
        val shoulderX = outputFeature0[SHOULDER_KEYPOINT_INDEX + 1]
        val shoulderY = outputFeature0[SHOULDER_KEYPOINT_INDEX]
        val hipX = outputFeature0[HIP_KEYPOINT_INDEX + 1]
        val hipY = outputFeature0[HIP_KEYPOINT_INDEX]

        val distanceBetweenShouldersAndHips = calculateDistance(shoulderX, shoulderY, hipX, hipY)

        // Adjust the threshold based on your specific criteria
        if (distanceBetweenShouldersAndHips < PROPORTION_THRESHOLD) {
            // Body proportions are good
            showProportionAssessment("Good body proportions!")
        } else {
            // Body proportions are not within the expected range
            showProportionAssessment("Body proportions need improvement.")
        }
    }

    private fun calculateDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        return Math.sqrt((x2 - x1).toDouble().pow(2) + (y2 - y1).toDouble().pow(2)).toFloat()
    }

    private fun showProportionAssessment(message: String) {
        // You can display the assessment result to the user, e.g., in a Toast or a dialog
        // For simplicity, let's display it in a Toast here
        showToast(message)
    }

    private fun showToast(message: String) {
        runOnUiThread {
            Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
        }
    }

    fun get_permissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA),101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0]!=PackageManager.PERMISSION_GRANTED)get_permissions()

        }
    //override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        //super.onActivityResult(requestCode, resultCode, data)

       //if(resultCode == Activity.RESULT_OK && resultCode == REQUEST_CODE && data != null){
            //imageView.setImageBitmap(data.extras!!.get("data") as Bitmap )
            //}
        //}

}