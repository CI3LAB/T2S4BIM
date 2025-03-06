using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.Apis.Auth.OAuth2;
using Google.Cloud.Speech.V1;

namespace BIM
{
    internal class SpeechRecognition
    {
        public static string Recognize(string fileName)
        {
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = "D:\\miniconda3\\envs\\pytorch\\python.exe",
                Arguments = "D:\\BIM-LM\\BIM\\interaction\\asr.py",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

            Process process = Process.Start(start);
            process.WaitForExit();

            string output = process.StandardOutput.ReadToEnd();
            return output;
        }
    }
}