using Autodesk.Revit.DB;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Data;
using Autodesk.Revit.DB.Visual;
using System.Windows.Input;
using Autodesk.Revit.UI;

namespace BIM
{
    internal class Utils
    {
        public static List<string> option_based_property = new List<string>()
        {
            "Wall Closure", "Glass Pane Material", "Glass Pane Color", "Sash Material", "Sash Color", "Level", "Orientation", "Wrapping at Inserts", "Wrapping at Ends",
            "Function", "Material", "Color", "Base Constraint", "Top Constraint", "Room Bounding", "Cross-Section", "Structural", "Structural Usage", "Right Support",
            "Left Support", "Middle Support", "Base Level", "Top Level", "Rafter Cut", "Rafter or Truss", "Shape", "Multistory Top Level", "Door Material", "Door Color",
            "Frame Material", "Frame Color", "Moves with Grids"
        };

        public static List<string> scalar_based_property = new List<string>()
        {
            "Height", "Width", "Window Inset", "Sill Height", "Head Height", "Base Offset", "Top Offset", "Unconnected Height", "Angle from Vertical",
            "Maximum Riser Height", "Minimum Tread Depth", "Minimum Run Width", "Right Lateral Offset", "Left Lateral Offset", "Middle Support Number",
            "Desired Stair Height", "Desired Number of Risers", "Actual Tread Depth", "Thickness", "Base Offset From Level", "Fascia Depth", "Slope",
            "Maximum Incline Length", "Max Slope", "Height Offset from Level", "Trim Projection Exterior", "Trim Projection Interior", "Trim Width",
            "Depth", "Offset Base", "Offset Top"
        };

        public static List<string> no_unit_scalar_based_property = new List<string>()
        {
            "Middle Support Number", "Desired Number of Risers", "Max Slope"
        };

        public static List<string> degree_based_property = new List<string>()
        {
            "Slope", "Angle from Vertical"
        };

        public static List<string> text_based_property = new List<string>()
        {
            "Type Comments", "Comments"
        };

        public static List<string> read_only_property = new List<string>()
        {
            "Coordinate", "Length", "Area", "Volume", "Actual Number of Risers", "Actual Riser Height", "Maximum Ridge Height"
        };

        public static double Mm2m(double value)
        {
            return value / 1000;
        }

        public static double M2mm(double value)
        {
            return value * 1000;
        }

        public static double Inch2mm(double inches)
        {
            double millimeters = inches * 25.4;
            return millimeters;
        }

        public static double Mm2inch(double millimeters)
        {
            double inches = millimeters / 25.4;
            return inches;
        }

        public static double Foot2mm(double feet)
        {
            double millimeters = feet * 304.8;
            return millimeters;
        }

        public static double Mm2foot(double millimeters)
        {
            double feet = millimeters / 304.8;
            return feet;
        }

        public static double Degree2radian(double degree)
        {
            double radian = degree * 0.0174533;
            return radian;
        }

        public static double Radian2degree(double radian)
        {
            double degree = radian / 0.0174533;
            return degree;
        }

        public static string RemoveNonNumericChars(string input)
        {
            string pattern = @"[^\d.-]";
            string ss = Regex.Replace(input, pattern, "");
            ss = RemoveExtraSymbol(RemoveInNumber(ss));
            if (string.IsNullOrEmpty(ss) || ss == "-" || ss == ".")
            {
                return "0";
            }
            return ss;
        }

        public static string RemoveExtraSymbol(string s)
        {
            if (s.Count(c => c == '-') > 1)
            {
                int first = s.IndexOf('-');
                s = s.Substring(0, first + 1) + s.Substring(first + 1).Replace("-", "");
            }
            if (s.Count(c => c == '.') > 1)
            {
                int first = s.IndexOf('.');
                s = s.Substring(0, first + 1) + s.Substring(first + 1).Replace(".", "");
            }
            return s;
        }

        public static string RemoveInNumber(string numberString)
        {
            if (string.IsNullOrEmpty(numberString))
            {
                return "0";
            }
            string firstChar = numberString[0].ToString();
            if (firstChar == ".")
            {
                firstChar = "0.";
            }
            string updatedString = firstChar + numberString.Substring(1).Replace("-", "");
            return updatedString;
        }

        public static string ScalarConvert(string value, string unit)
        {
            if (unit.ToLower().Contains("mm") || unit.ToLower().Contains("millimeter"))
            {
                return Mm2foot(double.Parse(RemoveNonNumericChars(value))).ToString();
            }
            else if (unit.ToLower().Contains("inch") || unit.ToLower().Contains("inches"))
            {
                return Mm2foot(Inch2mm(double.Parse(RemoveNonNumericChars(value)))).ToString();
            }
            else if (unit.ToLower().Contains("foot") || unit.ToLower().Contains("feet"))
            {
                return RemoveNonNumericChars(value);
            }
            else if (unit.ToLower().Contains("m") || unit.ToLower().Contains("meter"))
            {
                return Mm2foot(M2mm(double.Parse(RemoveNonNumericChars(value)))).ToString();
            }
            else
            {
                throw new ArgumentException("Invalid unit");
            }
        }

        public static KeyValuePair<string, string> MappingToRevit(string element, string property, string value, string unit)
        {
            List<string> remain = new List<string>() { "Level", "Comments", "Glass Pane Material", "Glass Pane Color", "Sash Material", "Sash Color", "Type Comments", "Base Constraint",
                                                       "Top Constraint", "Material", "Color", "Base Level", "Top Level", "Desired Number of Risers", "Middle Support Number", "Multistory Top Level",
                                                       "Door Material", "Door Color", "Frame Material", "Frame Color"
                                                     };
            List<string> scalar = new List<string>() { "Sill Height", "Head Height", "Height", "Width", "Window Inset", "Base Offset", "Top Offset", "Unconnected Height", "Desired Stair Height",
                                                       "Actual Tread Depth", "Maximum Riser Height", "Minimum Tread Depth", "Minimum Run Width", "Right Lateral Offset", "Left Lateral Offset",
                                                       "Base Offset From Level", "Fascia Depth", "Thickness", "Maximum Incline Length", "Height Offset from Level", "Trim Projection Exterior",
                                                       "Trim Projection Interior", "Trim Width", "Depth", "Offset Base", "Offset Top"
                                                     };
            if (element == "window")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Orientation")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Vertical", "0"},
                        {"Slanted", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Wall Closure")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"By host", "0"},
                        {"Neither", "1"},
                        {"Interior", "2"},
                        {"Exterior", "3"},
                        {"Both", "4"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "wall")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Room Bounding")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Cross-Section")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Slanted", "0"},
                        {"Vertical", "1"},
                        {"Tapered", "2"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Angle from Vertical")
                {
                    if (unit.ToLower().Contains("degree"))
                    {
                        return new KeyValuePair<string, string>(property, Degree2radian(double.Parse(value)).ToString());
                    }
                    else
                    {
                        return new KeyValuePair<string, string>(property, value);
                    }
                }
                else if (property == "Structural")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Structural Usage")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Non-Bearing", "0"},
                        {"Bearing", "1"},
                        {"Shear", "2"},
                        {"Structural combined", "3"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Wrapping at Inserts")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Do not wrap", "0"},
                        {"Exterior", "1"},
                        {"Interior", "2"},
                        {"Both", "3"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Wrapping at Ends")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"None", "0"},
                        {"Exterior", "1"},
                        {"Interior", "2"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Function")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Interior", "0"},
                        {"Exterior", "1"},
                        {"Foundation", "2"},
                        {"Retaining", "3"},
                        {"Soffit", "4"},
                        {"Core-shaft", "5"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "stair")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Function")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Interior", "0"},
                        {"Exterior", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Right Support")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"None", "0"},
                        {"Stringer - Closed", "1"},
                        {"Carriage - Open", "2"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Left Support")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"None", "0"},
                        {"Stringer - Closed", "1"},
                        {"Carriage - Open", "2"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Middle Support")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "roof")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Room Bounding")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Rafter Cut")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Plumb Cut", "33615"},
                        {"Two Cut - Plumb", "33619"},
                        {"Two Cut - Square", "33618"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Rafter or Truss")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Truss", "0"},
                        {"Rafter", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Slope")
                {
                    string new_value;
                    if (unit.Contains("radian"))
                    {
                        new_value = Radian2degree(double.Parse(value)).ToString();
                    }
                    else
                    {
                        new_value = value;
                    }
                    return new KeyValuePair<string, string>(property, Math.Tan(double.Parse(new_value)).ToString());
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "ramp")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Shape")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Thick", "0"},
                        {"Solid", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Function")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Interior", "0"},
                        {"Exterior", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "floor")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Room Bounding")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Structural")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Slope")
                {
                    string new_value;
                    if (unit.Contains("radian"))
                    {
                        new_value = Radian2degree(double.Parse(value)).ToString();
                    }
                    else
                    {
                        new_value = value;
                    }
                    return new KeyValuePair<string, string>(property, Math.Tan(double.Parse(new_value)).ToString());
                }
                else if (property == "Function")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Interior", "0"},
                        {"Exterior", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "door")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Orientation")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Vertical", "0"},
                        {"Slanted", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Wall Closure")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"By host", "0"},
                        {"Neither", "1"},
                        {"Interior", "2"},
                        {"Exterior", "3"},
                        {"Both", "4"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Function")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"Interior", "0"},
                        {"Exterior", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "ceiling")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Room Bounding")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Slope")
                {
                    string new_value;
                    if (unit.Contains("radian"))
                    {
                        new_value = Radian2degree(double.Parse(value)).ToString();
                    }
                    else
                    {
                        new_value = value;
                    }
                    return new KeyValuePair<string, string>(property, Math.Tan(double.Parse(new_value)).ToString());
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else if (element == "column")
            {
                if (remain.Contains(property))
                {
                    return new KeyValuePair<string, string>(property, value);
                }
                else if (scalar.Contains(property))
                {
                    string new_value = ScalarConvert(value, unit);
                    return new KeyValuePair<string, string>(property, new_value);
                }
                else if (property == "Moves with Grids")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else if (property == "Room Bounding")
                {
                    Dictionary<string, string> mapping = new Dictionary<string, string>()
                    {
                        {"No", "0"},
                        {"Yes", "1"}
                    };
                    return new KeyValuePair<string, string>(property, mapping[value]);
                }
                else
                {
                    throw new ArgumentException("Invalid element type");
                }
            }
            else
            {
                throw new ArgumentException("Invalid element type");
            }
        }

        public static List<KeyValuePair<string, string>> SwapManipulation(List<KeyValuePair<string, string>> property_value_pairs)
        {
            int color_index;
            int material_index;

            color_index = property_value_pairs.FindIndex(x => x.Key == "Color");
            material_index = property_value_pairs.FindIndex(x => x.Key == "Material");
            if (color_index != -1 && material_index != -1 && color_index < material_index)
            {
                (property_value_pairs[material_index], property_value_pairs[color_index]) = (property_value_pairs[color_index], property_value_pairs[material_index]);
            }

            color_index = property_value_pairs.FindIndex(x => x.Key == "Glass Pane Color");
            material_index = property_value_pairs.FindIndex(x => x.Key == "Glass Pane Material");
            if (color_index != -1 && material_index != -1 && color_index < material_index)
            {
                (property_value_pairs[material_index], property_value_pairs[color_index]) = (property_value_pairs[color_index], property_value_pairs[material_index]);
            }

            color_index = property_value_pairs.FindIndex(x => x.Key == "Sash Color");
            material_index = property_value_pairs.FindIndex(x => x.Key == "Sash Material");
            if (color_index != -1 && material_index != -1 && color_index < material_index)
            {
                (property_value_pairs[material_index], property_value_pairs[color_index]) = (property_value_pairs[color_index], property_value_pairs[material_index]);
            }

            color_index = property_value_pairs.FindIndex(x => x.Key == "Door Color");
            material_index = property_value_pairs.FindIndex(x => x.Key == "Door Material");
            if (color_index != -1 && material_index != -1 && color_index < material_index)
            {
                (property_value_pairs[material_index], property_value_pairs[color_index]) = (property_value_pairs[color_index], property_value_pairs[material_index]);
            }

            color_index = property_value_pairs.FindIndex(x => x.Key == "Frame Color");
            material_index = property_value_pairs.FindIndex(x => x.Key == "Frame Material");
            if (color_index != -1 && material_index != -1 && color_index < material_index)
            {
                (property_value_pairs[material_index], property_value_pairs[color_index]) = (property_value_pairs[color_index], property_value_pairs[material_index]);
            }

            return property_value_pairs;
        }

        public static Manipulation ParseManipulation(string json)
        {
            JObject jObject = JObject.Parse(json);
            string intent = jObject["intent"].ToString();
            string element = jObject["element"].ToString();
            List<KeyValuePair<string, string>> property_value_pairs = new List<KeyValuePair<string, string>>();

            if (intent == "deletion")
            {
                return new Manipulation(intent, element, property_value_pairs);
            }
            else if (intent == "retrieval")
            {
                foreach (var property in jObject["property"])
                {
                    string property_name = property["property"].ToString();
                    property_value_pairs.Add(new KeyValuePair<string, string>(property_name, ""));
                }
                return new Manipulation(intent, element, property_value_pairs);
            }
            else
            {
                foreach (var property in jObject["property"])
                {
                    string property_name = property["property"].ToString();
                    string value;
                    string unit;

                    if (property["target_value"] != null)
                    {
                        value = property["target_value"].ToString();
                        if (property["target_unit"] != null)
                        {
                            unit = property["target_unit"].ToString();
                        }
                        else
                        {
                            unit = "";
                        }
                        property_value_pairs.Add(MappingToRevit(element, property_name, value, unit));
                    }
                    else
                    {
                        value = property["value"].ToString();
                        if (property["unit"] != null)
                        {
                            unit = property["unit"].ToString();
                        }
                        else
                        {
                            unit = "";
                        }
                        property_value_pairs.Add(MappingToRevit(element, property_name, value, unit));
                    }
                }

                property_value_pairs = SwapManipulation(property_value_pairs);
                return new Manipulation(intent, element, property_value_pairs);
            }
        }

        public static int LevenshteinDistance(string str1, string str2)
        {
            int m = str1.Length;
            int n = str2.Length;
            int[,] L = new int[m + 1, n + 1];
            for (int i = 0; i <= m; i++)
            {
                for (int j = 0; j <= n; j++)
                {
                    if (i == 0 || j == 0)
                    {
                        L[i, j] = 0;
                    }
                    else if (str1[i - 1] == str2[j - 1])
                    {
                        L[i, j] = L[i - 1, j - 1] + 1;
                    }
                    else
                    {
                        L[i, j] = Math.Max(L[i - 1, j], L[i, j - 1]);
                    }
                }
            }
            int lcs = L[m, n];
            return (m - lcs) + (n - lcs);
        }

        public class FailuresPreprocessor : IFailuresPreprocessor
        {
            public FailureProcessingResult PreprocessFailures(FailuresAccessor failuresAccessor)
            {
                IList<FailureMessageAccessor> listFma = failuresAccessor.GetFailureMessages();
                if (listFma.Count == 0)
                    return FailureProcessingResult.Continue;
                foreach (FailureMessageAccessor fma in listFma)
                {
                    if (fma.GetSeverity() == FailureSeverity.Error)
                    {
                        if (fma.HasResolutions())
                            failuresAccessor.ResolveFailure(fma);
                    }
                    if (fma.GetSeverity() == FailureSeverity.Warning)
                    {
                        failuresAccessor.DeleteWarning(fma);
                    }
                }
                return FailureProcessingResult.ProceedWithCommit;
            }
        }
    }
}
