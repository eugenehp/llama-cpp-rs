package com.example.llama

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import kotlin.concurrent.thread

/**
 * Minimal demo: type a prompt, tap Generate, and run on-device inference through
 * [LlamaBridge] (which calls into `llama-cpp-4` via JNI).
 *
 * The model is read from `filesDir/model.gguf`. Push one with, e.g.:
 * ```
 * adb push stories260K.gguf /data/data/com.example.llama/files/model.gguf
 * ```
 */
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val promptField = findViewById<EditText>(R.id.prompt)
        val output = findViewById<TextView>(R.id.output)
        val runButton = findViewById<Button>(R.id.run)

        val modelFile = File(filesDir, "model.gguf")

        runButton.setOnClickListener {
            if (!modelFile.exists()) {
                output.text = getString(
                    R.string.missing_model,
                    modelFile.absolutePath,
                )
                return@setOnClickListener
            }

            runButton.isEnabled = false
            output.text = getString(R.string.generating)
            val prompt = promptField.text.toString()

            // Inference is blocking; keep it off the UI thread.
            thread {
                val result = try {
                    LlamaBridge.generate(modelFile.absolutePath, prompt, 48)
                } catch (t: Throwable) {
                    "error: ${t.message}"
                }
                runOnUiThread {
                    output.text = result
                    runButton.isEnabled = true
                }
            }
        }
    }
}
