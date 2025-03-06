using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.UI;
using Autodesk.Revit.UI.Selection;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Principal;
using System.Threading;

namespace BIM
{
    [TransactionAttribute(TransactionMode.Manual)]
    [RegenerationAttribute(RegenerationOption.Manual)]
    public class ElementOperation : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            UIDocument uidoc = commandData.Application.ActiveUIDocument;
            Document doc = uidoc.Document;
            
            var window = new UserControl();
            window.ShowDialog();
            string user_needs = "";

            if (window.flag==2){
                user_needs = window.user_needs.Text;
            }
            else
            {
                return Result.Succeeded;
            }

            System.IO.File.WriteAllText("D:\\BIM-LM\\BIM\\interaction\\request_input.txt", user_needs);

            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = "D:\\miniconda3\\envs\\pytorch\\python.exe",
                Arguments = "D:\\BIM-LM\\BIM\\interaction\\predict.py",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };
            Process.Start(start).WaitForExit();

            string response = System.IO.File.ReadAllText("D:\\BIM-LM\\BIM\\interaction\\SIF_response.json");

            Manipulation manipulation = Utils.ParseManipulation(response);

            if (manipulation.element == "window")
            {
                manipulation.WindowManipulation(uidoc, doc);
            }
            else if (manipulation.element == "wall")
            {
                manipulation.WallManipulation(uidoc, doc);
            }
            else if (manipulation.element == "stair")
            {
                manipulation.StairManipulation(uidoc, doc);
            }
            else if (manipulation.element == "roof")
            {
                manipulation.RoofManipulation(uidoc, doc);
            }
            else if (manipulation.element == "ramp")
            {
                manipulation.RampManipulation(uidoc, doc);
            }
            else if (manipulation.element == "floor")
            {
                manipulation.FloorManipulation(uidoc, doc);
            }
            else if (manipulation.element == "door")
            {
                manipulation.DoorManipulation(uidoc, doc);
            }
            else if (manipulation.element == "column")
            {
                manipulation.ColumnManipulation(uidoc, doc);
            }
            else if (manipulation.element == "ceiling")
            {
                manipulation.CeilingManipulation(uidoc, doc);
            }
            else
            {
                throw new System.Exception("Element not found");
            }
            
            return Result.Succeeded;
        }
    }
}
