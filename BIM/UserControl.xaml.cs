using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Autodesk.Revit.UI;
using NAudio.Wave;
using System.Threading;

namespace BIM
{
    /// <summary>
    /// Interaction logic for UserControl.xaml
    /// </summary>
    public partial class UserControl : Window
    {
        public int flag;
        SpeechRecorder recorder = null;

        public UserControl()
        {
            flag = 0;
            InitializeComponent();
            recorder = new SpeechRecorder();
        }

        private void Button_Click_Enter(object sender, RoutedEventArgs e)
        {
            flag = 2;
            this.Close();
        }

        private void Button_Click_Cancel(object sender, RoutedEventArgs e)
        {
            flag = 0;
            this.Close();
        }

        private void Button_Speak_Down(object sender, MouseButtonEventArgs e)
        {
            this.recorder.StartRecording();
        }
        
        private void Button_Speak_Up(object sender, MouseButtonEventArgs e)
        {
            this.recorder.StopRecording();
            flag = 1;
            string speech_text = SpeechRecognition.Recognize(recorder.fileName);
            this.user_needs.Text = speech_text;
        }
    }
}
