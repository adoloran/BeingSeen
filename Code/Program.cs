
using System.Collections.Generic;
using System;
using Microsoft.Psi;
using MathNet.Spatial.Euclidean;
using System.ComponentModel.Design;



namespace Test
    {
        class Program
        {

            static void Main(string[] args)
            {
                using (var p = Pipeline.Create(enableDiagnostics: true))
                {
                    Microsoft.Psi.Data.PsiImporter inputStore = PsiStore.Open(p, "Faceopen", @"C:\Users\Jarvis\Desktop\STAGE");

                    var csSource = inputStore.OpenStream<CoordinateSystem>("gazeout");
                    var csv_lines = new List<string>();
                    csSource.Do(m =>
                    {
                        csv_lines.Add($"first_column,{string.Join(",", m.Storage.ToRowMajorArray())}");
                    });

                    p.Run(ReplayDescriptor.ReplayAll);

                    System.IO.File.WriteAllLines("test.csv", csv_lines);
                }
            }
        }
    }
