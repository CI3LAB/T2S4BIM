using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.UI;
using Autodesk.Revit.UI.Selection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BIM
{
    internal class Manipulation
    {
        public string intent;
        public string element;
        public List<KeyValuePair<string, string>> property_value_pairs;

        public Manipulation(string intent, string element, List<KeyValuePair<string, string>> property_value_pairs)
        {
            this.intent = intent;
            this.element = element;
            this.property_value_pairs = property_value_pairs;
        }

        public void WindowManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a window", "Please click where you need to create the window");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                WindowFunc.Create(doc, point1, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a window", "Please click on the window you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance window = element as FamilyInstance;
                WindowFunc.Delete(window);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a window", "Please click on the window you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance window = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    WindowFunc.Retrieval(window, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a window", "Please click on the window you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance window = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    WindowFunc.Modify(window, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }
        
        public void WallManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a wall", "Please click on two points as the bottom line to create the wall");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                WallFunc.Create(doc, point1, point2, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a wall", "Please click on the wall you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Wall wall = element as Wall;
                WallFunc.Delete(wall);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a wall", "Please click on the wall you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Wall wall = element as Wall;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    WallFunc.Retrieval(wall, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a wall", "Please click on the wall you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Wall wall = element as Wall;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    WallFunc.Modify(wall, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }

        public void StairManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a stair", "Please click on two points as the central line to create the stair");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                StairFunc.Create(doc, point1, point2, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a stair", "Please click on the stair you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Stairs stair = element as Stairs;
                StairFunc.Delete(stair);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a stair", "Please click on the stair you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Stairs stair = element as Stairs;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    StairFunc.Retrieval(stair, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a stair", "Please click on the stair you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Stairs stair = element as Stairs;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    StairFunc.Modify(stair, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }

        public void RoofManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a roof", "Please click on four points as the outline to create the roof");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                XYZ point3 = sel.PickPoint();
                XYZ point4 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                RoofFunc.Create(doc, point1, point2, point3, point4, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a roof", "Please click on the roof you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FootPrintRoof roof = element as FootPrintRoof;
                RoofFunc.Delete(roof);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a roof", "Please click on the roof you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FootPrintRoof roof = element as FootPrintRoof;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    RoofFunc.Retrieval(roof, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a roof", "Please click on the roof you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FootPrintRoof roof = element as FootPrintRoof;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    RoofFunc.Modify(roof, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }

        public void RampManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a ramp", "Please click on two points as the central line to create the ramp");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                RampFunc.Create(doc, point1, point2, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a ramp", "Please click on the ramp you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Element ramp = element;
                RampFunc.Delete(ramp);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a ramp", "Please click on the ramp you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Element ramp = element;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    RampFunc.Retrieval(ramp, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a ramp", "Please click on the ramp you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Element ramp = element;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    RampFunc.Modify(ramp, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }

        public void FloorManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a floor", "Please click on four points as the outline to create the floor");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                XYZ point3 = sel.PickPoint();
                XYZ point4 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                FloorFunc.Create(doc, point1, point2, point3, point4, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a floor", "Please click on the floor you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Floor floor = element as Floor;
                FloorFunc.Delete(floor);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a floor", "Please click on the floor you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Floor floor = element as Floor;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    FloorFunc.Retrieval(floor, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a floor", "Please click on the floor you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Floor floor = element as Floor;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    FloorFunc.Modify(floor, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }
    
        public void DoorManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a door", "Please click where you need to create the door");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                DoorFunc.Create(doc, point1, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a door", "Please click on the door you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance door = element as FamilyInstance;
                DoorFunc.Delete(door);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a door", "Please click on the door you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance door = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    DoorFunc.Retrieval(door, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a door", "Please click on the door you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance door = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    DoorFunc.Modify(door, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }
    
        public void ColumnManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a column", "Please click where you need to create the column");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                ColumnFunc.Create(doc, point1, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a column", "Please click on the column you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance column = element as FamilyInstance;
                ColumnFunc.Delete(column);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a column", "Please click on the column you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance column = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    ColumnFunc.Retrieval(column, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a column", "Please click on the column you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                FamilyInstance column = element as FamilyInstance;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    ColumnFunc.Modify(column, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }
    
        public void CeilingManipulation(UIDocument uidoc, Document doc)
        {
            if (this.intent == "creation")
            {
                TaskDialog.Show("Create a ceiling", "Please click on four points as the outline to create the ceiling");
                Selection sel = uidoc.Selection;
                XYZ point1 = sel.PickPoint();
                XYZ point2 = sel.PickPoint();
                XYZ point3 = sel.PickPoint();
                XYZ point4 = sel.PickPoint();
                Dictionary<string, string> dic = new Dictionary<string, string>();
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    dic.Add(pair.Key, pair.Value);
                }
                CeilingFunc.Create(doc, point1, point2, point3, point4, dic);
            }
            else if (this.intent == "deletion")
            {
                TaskDialog.Show("Delete a ceiling", "Please click on the ceiling you need to remove");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Ceiling ceiling = element as Ceiling;
                CeilingFunc.Delete(ceiling);
            }
            else if (this.intent == "retrieval")
            {
                TaskDialog.Show("Retrieve a ceiling", "Please click on the ceiling you need to retrieve");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Ceiling ceiling = element as Ceiling;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    CeilingFunc.Retrieval(ceiling, property_name);
                }
            }
            else if (this.intent == "modification")
            {
                TaskDialog.Show("Modify a ceiling", "Please click on the ceiling you need to modify");
                Selection sel = uidoc.Selection;
                Reference r = sel.PickObject(ObjectType.PointOnElement);
                Element element = doc.GetElement(r);
                Ceiling ceiling = element as Ceiling;
                foreach (KeyValuePair<string, string> pair in this.property_value_pairs)
                {
                    string property_name = pair.Key;
                    string property_value = pair.Value;
                    CeilingFunc.Modify(ceiling, property_name, property_value);
                }
            }
            else
            {
                throw new Exception("Invalid intent");
            }
        }
    }
}
