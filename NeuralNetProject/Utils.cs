using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetProject
{
    public static class Utils
    {

        public static float Sigmoid(float _s, float _beta)
        {
            return 1 / (1 + MathF.Exp(-_beta * _s));
        }

        public static float dSigmoid(float _s, float _beta)
        {
            return _beta * Sigmoid(_s, _beta) * (1 - Sigmoid(_s, _beta));
        }

        public static float Error(float[] output, float[] expected)
        {
            float cost = 0;
            for (int i = 0; i < output.Length; i++) cost += (float)Math.Pow(output[i] - expected[i], 2);
            return cost / 2;
        }
        public static float RandomRange(float a, float b)
        {
            return (float)new Random().NextDouble() * (b - a) + a;
        }


    }
}
