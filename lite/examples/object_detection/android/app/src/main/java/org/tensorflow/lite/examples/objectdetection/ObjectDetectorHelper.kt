/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF
import org.tensorflow.lite.DataType

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder


import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.core.CvType
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import org.opencv.core.Core



class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null

    private var customYoloDetector: Detector? = null
    private var isUsingYoloV8 = false
    private var isUsingYoloV8Int8 = false
    private var labels: List<String> = emptyList()
    private var lastFrameTime: Long = 0
    //MOG 추가
    private val mog2 = Video.createBackgroundSubtractorMOG2()

    init {
        setupObjectDetector()


    }

    fun clearObjectDetector() {
        objectDetector = null
        customYoloDetector = null
        isUsingYoloV8 = false
        isUsingYoloV8Int8 = false
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }

            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }

            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }

        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                MODEL_YOLOV8 -> "model-yolov8.tflite"
                MODEL_YOLOV8_INT8 -> "yolov8n-full-int8.tflite"
                else -> "mobilenetv1.tflite"
            }

        isUsingYoloV8 = (currentModel == MODEL_YOLOV8)
        isUsingYoloV8Int8 = (currentModel == MODEL_YOLOV8_INT8)

        if (isUsingYoloV8 || isUsingYoloV8Int8) {
            try {
                labels = loadLabels("labels.txt")
                customYoloDetector = Detector(context, modelName, "labels.txt", object : Detector.DetectorListener {
                    override fun onEmptyDetect() {
                        val fps = calculateFps()
                        objectDetectorListener?.onResults(
                            results = mutableListOf(),
                            inferenceTime = 0,
                            fps = fps,
                            imageHeight = 0,
                            imageWidth = 0,
                            labels = labels
                        )
                    }



                    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
                        // 변환: BoundingBox → Detection 형식
                        val convertedResults = boundingBoxes.map { box ->
                            Detection.create(
                                RectF(box.x1, box.y1, box.x2, box.y2),
                                listOf(Category.create(box.clsName, box.clsName, box.cnf))
                            )
                        }.toMutableList()
                        val fps = calculateFps()

                        objectDetectorListener?.onResults(
                            convertedResults,
                            inferenceTime,
                            fps,
                            1, // YOLO는 정규화된 좌표 사용 → Overlay에서 조정
                            1,
                            labels
                        )
                    }
                })
            } catch (e: Exception) {
                objectDetectorListener?.onError("YOLOv8 초기화 실패: ${e.message}")
                Log.e("ObjectDetectorHelper", "YOLOv8 Detector init error", e)
            }

        } else {
            try {
                objectDetector =
                    ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
            } catch (e: IllegalStateException) {
                objectDetectorListener?.onError(
                    "Object detector failed to initialize. See error logs for details"
                )
                Log.e("Test", "TFLite failed to load model with error: " + e.message)
            }

            }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null && customYoloDetector == null) {
            setupObjectDetector()
        }

        when {
            isUsingYoloV8Int8 -> {
                detectYoloV8INT8()
                return
            }
            isUsingYoloV8 -> {
                detectYoloV8FP32(image)
                return
            }
        }

        // OpenCV MOG2 전처리
        val mat = Mat()
        Utils.bitmapToMat(image, mat)

        val fgMask = Mat()
        mog2.apply(mat, fgMask)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(fgMask, fgMask, Imgproc.MORPH_OPEN, kernel)

        val foreground = Mat()
        Core.bitwise_and(mat, mat, foreground, fgMask)

        val processedBitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(foreground, processedBitmap)

        var inferenceTime = SystemClock.uptimeMillis()

        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .build()

        // 🔥 반드시 전처리된 processedBitmap 사용
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(processedBitmap))

        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        val fps = calculateFps()

        objectDetectorListener?.onResults(
            results,
            inferenceTime,
            fps,
            tensorImage.height,
            tensorImage.width,
            labels
        )
    }


    private fun loadLabels(path: String): List<String> {
        return context.assets.open(path).bufferedReader().useLines { it.toList() }
    }


    fun byteBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        buffer.rewind()

        for (i in 0 until width * height) {
            val r = buffer.get().toInt() and 0xFF
            val g = buffer.get().toInt() and 0xFF
            val b = buffer.get().toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b  // ARGB
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }


    // YOLOv8 FP32 처리
    private fun detectYoloV8FP32(image: Bitmap) {
        customYoloDetector?.detect(image)
    }


    // YOLOv8 INT8 처리 (흰색 이미지 ByteBuffer 생성)
    private fun detectYoloV8INT8() {
        val width = 224
        val height = 224

        val inputBuffer = ByteBuffer.allocateDirect(width * height * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        repeat(width * height) {
            inputBuffer.put(255.toByte()) // R
            inputBuffer.put(255.toByte()) // G
            inputBuffer.put(255.toByte()) // B
        }
        inputBuffer.rewind()
        val bitmap = byteBufferToBitmap(inputBuffer, 224, 224)
        customYoloDetector?.detect(bitmap)
    }


    private fun calculateFps(): Float {
        val currentTime = SystemClock.elapsedRealtime()
        val fps = if (lastFrameTime > 0) {
            1000f / (currentTime - lastFrameTime).coerceAtLeast(1)
        } else {
            0f
        }
        lastFrameTime = currentTime
        return fps
    }


    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: MutableList<Detection>?,
          inferenceTime: Long,
          fps: Float,
          imageHeight: Int,
          imageWidth: Int,
          labels: List<String>
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_YOLOV8 = 4
        const val MODEL_YOLOV8_INT8 = 5
    }
}



