package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.Intent
import android.content.Context
import android.provider.Settings
import android.net.Uri
import kotlinx.coroutines.*
import androidx.lifecycle.lifecycleScope
import java.io.FileWriter
import org.json.JSONArray
import java.io.InputStream
import java.io.OutputStream
import okhttp3.*
import android.provider.MediaStore
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import android.util.Log
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import okhttp3.Request
import okhttp3.Response



class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var resultTextView: TextView
    private lateinit var testresultTextView: TextView
    private lateinit var progressTextView: TextView
    private lateinit var imageView: ImageView
    private lateinit var classNames: Map<Int, String>

    data class PredictionResult(
        val imagePath: String,
        val trueClass: String,
        val predictedClass: String,
        val isCorrect: Boolean
    )

    private val predictionResults = mutableListOf<PredictionResult>()

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)

        .build()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultTextView = findViewById(R.id.resultTextView)
        imageView = findViewById(R.id.imageView)
        testresultTextView = findViewById(R.id.testresultTextView)
        progressTextView = findViewById(R.id.progressTextView)


        // TFLite 모델 로드
        interpreter = Interpreter(loadModelFile())

        // class 이름 맵 로드
        classNames = loadClassNames(this)

        // 이미지 선택 버튼
        val selectImageButton: Button = findViewById(R.id.selectImageButton)
        selectImageButton.setOnClickListener {
            openImagePicker()
        }

        // 정확도 테스트 버튼
        val testButton: Button = findViewById(R.id.testButton)
        testButton.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (!Environment.isExternalStorageManager()) {
                    // 설정 화면으로 유도
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.data = Uri.parse("package:$packageName")
                    startActivity(intent)
                } else {
                    testModelAccuracy()
                }
            } else {
                if (ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.READ_EXTERNAL_STORAGE
                    ) != PackageManager.PERMISSION_GRANTED
                ) {
                    ActivityCompat.requestPermissions(
                        this,
                        arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                        REQUEST_CODE_READ_STORAGE
                    )
                } else {
                    testModelAccuracy()
                }
            }
        }

        // 이미지 전송 버튼
        val uploadImageButton: Button = findViewById(R.id.uploadImageButton)
        uploadImageButton.setOnClickListener {
            openSendImagePicker()
        }

        val downloadModelButton: Button = findViewById(R.id.downloadModelButton)
        downloadModelButton.setOnClickListener {
            downloadModelFromServer()
        }



    }


    // 권한 요청 결과 콜백
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_READ_STORAGE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                testModelAccuracy()
            } else {
                Toast.makeText(this, "외부 저장소 접근 권한이 필요합니다.", Toast.LENGTH_LONG).show()
            }
        }
    }

    // 모델 파일을 assets에서 로드
    private fun loadModelFile(): MappedByteBuffer {
        try {
            val fd = assets.openFd("downloaded_model.tflite")
            val inputStream = FileInputStream(fd.fileDescriptor)
            val channel = inputStream.channel
            val startOffset = fd.startOffset
            val declaredLength = fd.declaredLength
            return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: IOException) {
            throw RuntimeException("모델 파일 로드 실패", e)
        }
    }


    private val imagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val imageUri = result.data?.data
                imageUri?.let {
                    val bmp = BitmapFactory.decodeStream(contentResolver.openInputStream(it))
                    imageView.setImageBitmap(bmp)
                    bmp?.let { b -> runPrediction(b) }
                }
            }
        }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        imagePickerLauncher.launch(intent)
    }

    // 사용자 선택 이미지에 대한 예측
    private fun runPrediction(bitmap: Bitmap) {
        val inputArray = preprocessImage(bitmap)
        val outputSize = interpreter.getOutputTensor(0).shape()[1]
        val outputArray = Array(1) { FloatArray(outputSize) }

        interpreter.run(inputArray, outputArray)

        val top3 = getTop3Predictions(outputArray[0])
        displayResults(top3)
    }

    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                input[0][y][x][0] = Color.red(pixel) / 255.0f
                input[0][y][x][1] = Color.green(pixel) / 255.0f
                input[0][y][x][2] = Color.blue(pixel) / 255.0f
            }
        }
        return input
    }

    private fun getTop3Predictions(pred: FloatArray): List<Pair<Int, Float>> {
        val sorted = pred.indices.sortedByDescending { pred[it] }
        return sorted.take(3).map { it to pred[it] }
    }

    private fun displayResults(top3: List<Pair<Int, Float>>) {
        val sb = StringBuilder("Top 3 Predictions:\n")
        for ((idx, conf) in top3) {
            val name = classNames[idx] ?: "Unknown"
            sb.append("$name: ${"%.2f".format(conf * 100)}%\n")
        }
        resultTextView.text = sb.toString()
    }

    private fun testModelAccuracy() {
        lifecycleScope.launch(Dispatchers.IO) {
            var correct = 0
            var total = 0

            val testRoot = File(Environment.getExternalStorageDirectory(), "DCIM/test_new")
            if (!testRoot.exists() || !testRoot.isDirectory) {
                withContext(Dispatchers.Main) {
                    testresultTextView.text = "테스트 폴더가 존재하지 않습니다: ${testRoot.absolutePath}"
                }
                return@launch
            }

            val classFolders = testRoot.listFiles()?.filter { it.isDirectory } ?: emptyList()
            val totalClasses = classFolders.size

            val tempResults = mutableListOf<PredictionResult>()

            classFolders.forEachIndexed { classIdx, classFolder ->
                val actualClassName = classFolder.name
                val imageFiles = classFolder.listFiles()?.filter {
                    val ext = it.extension.lowercase()
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                } ?: emptyList()

                imageFiles.forEachIndexed { imgIdx, imgFile ->
                    val bitmap = BitmapFactory.decodeFile(imgFile.absolutePath)
                    val predIdx = runPredictionAndGetClass(bitmap)
                    val predName = classNames[predIdx] ?: "Unknown"
                    val isCorrect = predName == actualClassName

                    tempResults.add(PredictionResult(
                        imagePath = imgFile.absolutePath,
                        trueClass = actualClassName,
                        predictedClass = predName,
                        isCorrect = isCorrect
                    ))

                    if (isCorrect) {
                        correct++
                    }

                    total++

                    val currentAccuracy = (correct.toFloat() / total) * 100

                    // 진행 상황 업데이트
                    withContext(Dispatchers.Main) {
                        progressTextView.text = """
                        클래스: $actualClassName ($classIdx/${totalClasses})
                        이미지: ${imgIdx + 1}/${imageFiles.size}
                        정답 수: $correct / $total
                        현재 정확도: ${"%.2f".format(currentAccuracy)}%
                        """.trimIndent()
                    }
                }
            }


            val wrongResults = tempResults.filter { !it.isCorrect }

            // 오답 이미지 클래스별 폴더 생성
            val baseFolder = File(Environment.getExternalStorageDirectory(), "DCIM/wrong_images")
            if (!baseFolder.exists()) {
                baseFolder.mkdirs()
            }

            // 오답 이미지 클래스별로 폴더 생성 후 복사
            wrongResults.forEach { result ->
                val classFolder = File(baseFolder, result.trueClass) // 클래스명 폴더
                if (!classFolder.exists()) {
                    classFolder.mkdirs()
                }

                val sourceFile = File(result.imagePath)
                val destFile = File(classFolder, sourceFile.name)
                copyFile(sourceFile, destFile)  // 이미지 복사
            }


            savePredictionResults(tempResults)

            val accuracy = if (total > 0) (correct.toFloat() / total) * 100 else 0f

            withContext(Dispatchers.Main) {
                testresultTextView.text = "총 이미지: $total\n정답: $correct\n정확도: ${"%.2f".format(accuracy)}%"
                progressTextView.text = "테스트 완료"
            }
        }
    }

    private fun runPredictionAndGetClass(bitmap: Bitmap): Int {
        val inputArray = preprocessImage(bitmap)
        val outputSize = interpreter.getOutputTensor(0).shape()[1]
        val outputArray = Array(1) { FloatArray(outputSize) }
        interpreter.run(inputArray, outputArray)
        return outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
    }

    private fun savePredictionResults(results: List<PredictionResult>) {
        val metadataFile = File(applicationContext.filesDir, "prediction_results.json")
        val writer = FileWriter(metadataFile, true)

        results.forEach { result ->
            val jsonResult = JSONObject().apply {
                put("imagePath", result.imagePath)
                put("trueClass", result.trueClass)
                put("predictedClass", result.predictedClass)
                put("isCorrect", result.isCorrect)
            }

            writer.append(jsonResult.toString()).append("\n")
        }

        writer.close()
    }

    private fun copyFile(sourceFile: File, destFile: File) {
        try {
            if (!destFile.exists()) {
                destFile.createNewFile()
            }

            val inputStream: InputStream = sourceFile.inputStream()
            val outputStream: OutputStream = destFile.outputStream()

            val buffer = ByteArray(1024)
            var length: Int
            while (inputStream.read(buffer).also { length = it } > 0) {
                outputStream.write(buffer, 0, length)
            }

            inputStream.close()
            outputStream.close()

        } catch (e: IOException) {
            e.printStackTrace()
        }
    }






    private val sendImagePickerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val imageUri = result.data?.data
            imageUri?.let {
                try {
                    val inputStream = contentResolver.openInputStream(it)
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    if (bitmap != null) {
                        imageView.setImageBitmap(bitmap)
                        uploadImageToServer(it)
                    } else {
                        Toast.makeText(this, "이미지 로딩 실패", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    Toast.makeText(this, "이미지 디코딩 오류", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun openSendImagePicker() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        sendImagePickerLauncher.launch(intent)
    }


    private fun uploadImageToServer(imageUri: Uri) {
        val contentResolver = contentResolver

        try {
            // 서버 URL
            val url = "http://10.0.2.2:5000/upload_image"

            // InputStream을 사용하여 파일 전송
            val inputStream = contentResolver.openInputStream(imageUri) ?: return

            // MultipartBody로 이미지 전송 준비
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image", "uploaded_image.jpg",  // "image"는 서버에서 받을 필드명, "uploaded_image.jpg"는 파일명
                    inputStream.use { inputStream ->
                        inputStream.readBytes().toRequestBody("image/jpeg".toMediaTypeOrNull())  // InputStream -> RequestBody로 변환
                    }
                )
                .build()

            // 요청 준비
            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            // 비동기 요청 실행
            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {
                        Log.e("ImageUploadError", "이미지 업로드 실패: ${e.message}", e)
                        Toast.makeText(applicationContext, "이미지 업로드 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    if (response.isSuccessful) {
                        runOnUiThread {
                            Toast.makeText(applicationContext, "이미지 업로드 성공", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        runOnUiThread {
                            Toast.makeText(applicationContext, "이미지 업로드 실패", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            })
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                Toast.makeText(applicationContext, "이미지 처리 오류", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun downloadModelFromServer() {
        // 다운로드 URL (서버 URL을 사용)
        val actualIpAddress = "http://172.18.217.178" // <-- PC의 실제 IP
        val modelUrl = "http://${actualIpAddress}:5000/get_model"


        // 다운로드 시작
        val request = Request.Builder().url(modelUrl).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    // UI 스레드에서 Toast 메시지 출력
                    Toast.makeText(applicationContext, "모델 다운로드 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                if (response.isSuccessful) {
                    // 다운로드한 데이터 처리

                    val downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val file = File(downloadDir, "downloaded_model.tflite")

                    try {

                        response.body?.byteStream()?.use { inputStream ->
                            FileOutputStream(file).use { outputStream ->
                                val buffer = ByteArray(8192)  // 버퍼 크기 설정 (8KB)
                                var bytesRead: Int

                                // 파일을 스트리밍 방식으로 읽고 씀
                                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                                    outputStream.write(buffer, 0, bytesRead)
                                }
                            }
                        }

                        runOnUiThread {
                            Toast.makeText(applicationContext, "모델 다운로드 완료", Toast.LENGTH_SHORT).show()
                        }

                    } catch (e: IOException) {
                        e.printStackTrace()
                        runOnUiThread {
                            Toast.makeText(applicationContext, "파일 저장 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(applicationContext, "모델 다운로드 실패: 서버 오류", Toast.LENGTH_SHORT).show()
                    }
                }
            }
                }
        })
    }



    fun loadClassNames(context: Context, fileName: String = "new_class_names.json"): Map<Int, String> {
        val jsonStr = context.assets.open(fileName).bufferedReader().use { it.readText() }
        val obj = JSONObject(jsonStr)
        val map = mutableMapOf<Int, String>()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key.toInt()] = obj.getString(key)
        }
        return map
    }

    companion object {
        private const val REQUEST_CODE_READ_STORAGE = 100
    }
}
