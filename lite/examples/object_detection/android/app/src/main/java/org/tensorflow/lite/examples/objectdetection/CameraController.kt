package org.tensorflow.lite.examples.objectdetection

import android.hardware.camera2.*
import android.util.Log

class CameraController(
    private val cameraDevice: CameraDevice,
    private val captureSession: CameraCaptureSession
) {
    fun applySettings(iso: Int, shutter: Long) {
        val builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        if (iso > 0) builder.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
        if (shutter > 0) builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutter)
        captureSession.setRepeatingRequest(builder.build(), null, null)
        Log.d("CameraController", "Applied ISO=$iso, Shutter=$shutter")
    }
}