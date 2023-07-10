using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetProject
{
    class Program
    {
        static void Main(string[] args)
        {
            int maxFScount = 0;
            Random random = new Random();
            Program program = new Program();
            Console.WriteLine("Trwa uczenie...");
            NeuralNet net = new NeuralNet(new int[] {3, 10, 11});
            List<(float[], float[])> fs = new List<(float[], float[])>();
            List<(float[], float[])> sn = new List<(float[], float[])>();
            List<float[]> tests = new List<float[]>();
            using (var reader = new StreamReader(@"C:\Users\user\Desktop\data.csv"))
            {
                float[] input;
                float[] expected;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(';');
                    input = new float[3] { float.Parse(values[0]), float.Parse(values[1]), float.Parse(values[2]) };
                    expected = new float[11] { float.Parse(values[3]), float.Parse(values[4]), float.Parse(values[5]),
                                               float.Parse(values[6]), float.Parse(values[7]), float.Parse(values[8]),
                                               float.Parse(values[9]), float.Parse(values[10]), float.Parse(values[11]),
                                               float.Parse(values[12]), float.Parse(values[13])};
                    fs.Add((input,expected));
                }
                maxFScount = fs.Count;
            }
            while(sn.Count < 0.8f*maxFScount)
            {
                int index = random.Next(0, fs.Count);
                sn.Add(fs[index]);
                fs.RemoveAt(index);
            }

            net.Train(sn, 3000);
            Console.Clear();
            Console.WriteLine(net.output[net.output.Count-1]);
            Console.WriteLine("Testowy");
            program.TestAccuracy(fs, net);
            Console.WriteLine("Uczacy");
            program.TestAccuracy(sn, net);
            tests.Add(new float[] { 255, 0, 0 });
            tests.Add(new float[] { 0, 255, 0 });
            tests.Add(new float[] { 0, 0, 255 });
            tests.Add(new float[] { 255, 255, 255 });
            tests.Add(new float[] { 167, 0, 0 });
            tests.Add(new float[] {0, 0, 0 });

            foreach (float[] test in tests)
            {
                float[] results = net.Run(test);
                float maxValue = results.Max();
                int maxIndex = results.ToList().IndexOf(maxValue);
                Console.WriteLine($"Podany kolor [R:{test[0]}, G:{test[1]}, B:{test[2]}] to: " + ((Colors)maxIndex + 1) + $" [{maxValue}] ");
                Console.WriteLine("Dokladne wartości:");
                int f = 0;
                foreach (float result in results)
                {
                    Console.WriteLine("\t" + (Colors)(f+1) + ": " + result);
                    f++;
                }
            }

        }

        public void TestAccuracy(List<(float[], float[])> _fs, NeuralNet _net)
        {
            int goodResults = 0;
            foreach((float[],float[]) line in _fs)
            {
                float[] results = _net.Run(line.Item1);
                float maxValue = results.Max();
                int maxIndex = results.ToList().IndexOf(maxValue);
                if(maxIndex == line.Item2.ToList().IndexOf(line.Item2.Max()))
                {
                    goodResults++;
                }
            }
            Console.WriteLine($"Accuracy: {(float)((float)goodResults / _fs.Count) * 100}%");
        }

    }

    public enum Colors
    {
        Czerwony = 1,
        Zielony = 2,
        Niebieski = 3,
        Zolty = 4,
        Pomaranczowy = 5,
        Rozowy = 6,
        Fioletowy = 7,
        Brazowy = 8,
        Szary = 9,
        Czarny = 10,
        Bialy = 11
    }

}
