using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIM
{
    internal class SpeechRecorder
    {
        public WaveIn waveSource = null;
        public WaveFileWriter waveFile = null;
        public string fileName = "D:\\BIM-LM\\BIM\\interaction\\speech_input.wav";

        public void SetFileName(string fileName)
        {
            this.fileName = fileName;
        }

        public void StartRecording()
        {
            // plugging a microphone before invoking this function
            try
            {
                waveSource = new WaveIn();
                waveSource.WaveFormat = new WaveFormat(24000, 16, 1); // 24KHz, 16bit, mono

                waveSource.DataAvailable += new EventHandler<WaveInEventArgs>(WaveSource_DataAvailable);
                waveSource.RecordingStopped += new EventHandler<StoppedEventArgs>(WaveSource_RecordingStopped);

                waveFile = new WaveFileWriter(fileName, waveSource.WaveFormat);

                waveSource.StartRecording();
            }
            catch (Exception e)
            {
                throw new Exception(e.Message);
            }

        }

        public void StopRecording()
        {
            waveSource.StopRecording();

            if (waveSource != null)
            {
                waveSource.Dispose();
                waveSource = null;
            }

            if (waveFile != null)
            {
                waveFile.Dispose();
                waveFile = null;
            }
        }

        private void WaveSource_DataAvailable(object sender, WaveInEventArgs e)
        {
            if (waveFile != null)
            {
                waveFile.Write(e.Buffer, 0, e.BytesRecorded);
                waveFile.Flush();
            }
        }

        private void WaveSource_RecordingStopped(object sender, StoppedEventArgs e)
        {
            if (waveSource != null)
            {
                waveSource.Dispose();
                waveSource = null;
            }

            if (waveFile != null)
            {
                waveFile.Dispose();
                waveFile = null;
            }
        }
    }
}
