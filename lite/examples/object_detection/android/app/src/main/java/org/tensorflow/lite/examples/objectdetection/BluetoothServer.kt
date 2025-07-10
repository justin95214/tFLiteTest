package org.tensorflow.lite.examples.objectdetection

import android.util.Log
import kotlinx.coroutines.*
import org.json.JSONObject
import android.hardware.camera2.*

import androidx.core.content.ContextCompat
import android.Manifest
import android.content.pm.PackageManager
import android.bluetooth.BluetoothAdapter
import android.content.Context
import java.util.UUID
import java.io.OutputStream
import android.bluetooth.BluetoothSocket


// 필요하다면 Context 등 주입
// BluetoothServer.kt
class BluetoothServer(
    private val context: Context

) : CoroutineScope by MainScope() {

    private var socket: BluetoothSocket? = null
    private var outputStream: OutputStream? = null

    fun startListening() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("BT", "No BLUETOOTH_CONNECT permission")
            return
        }

        launch(Dispatchers.IO) {
            try {
                val adapter = BluetoothAdapter.getDefaultAdapter()
                val serverSocket = adapter.listenUsingRfcommWithServiceRecord(
                    "BT_APP",
                    UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
                )
                Log.d("BT", "Listening for BT connection...")

                socket = serverSocket.accept()
                outputStream = socket?.outputStream
                Log.d("BT", "Client connected!")

                val inputStream = socket?.inputStream
                val buffer = ByteArray(1024)

                while (true) {
                    val bytesRead = inputStream?.read(buffer) ?: -1
                    if (bytesRead > 0) {
                        val msg = String(buffer, 0, bytesRead)
                        Log.d("BT", "Received: $msg")
                    }
                }
            } catch (e: Exception) {
                Log.e("BT", "Error: $e")
            }
        }
    }

    fun sendMessage(msg: String) {
        try {
            outputStream?.write(msg.toByteArray())
            Log.d("BluetoothServer", "Sent: $msg")
        } catch (e: Exception) {
            Log.e("BluetoothServer", "Send failed: $e")
        }
    }


    fun stop() {
        try {
            socket?.close()
            Log.d("BT", "Bluetooth server stopped.")
        } catch (e: Exception) {
            Log.e("BT", "Close failed: $e")
        }
    }

}

