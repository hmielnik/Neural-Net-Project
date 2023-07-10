using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NeuralNetProject
{
    class NeuralNet
    {
        public int n = 1;
        public float beta = 1f;
        int[] h;
        float[][] v;
        float[][][] w;
        public List<float> output = new List<float>();
        ActivationFunctions activation;

        public NeuralNet(int[] _h, ActivationFunctions _activation = ActivationFunctions.Sigmoid)
        {
            h = new int[_h.Length];
            for (int i = 0; i < _h.Length; i++)
            {
                h[i] = _h[i];
            }
            activation = _activation;
            InitV();
            InitWeights();
        }

        public float GetGamma(int _n)
        {
            //return 1 / MathF.Pow(_n, .25f);
            return 0.4f;
        }

        private void InitV()
        {
            List<float[]> neuronsList = new List<float[]>();
            for (int i = 0; i < h.Length; i++)
            {
                neuronsList.Add(new float[h[i]]);
            }
            v = neuronsList.ToArray();
        }

        private void InitWeights()
        {
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < h.Length; i++)
            {
                List<float[]> layerWeightsList = new List<float[]>();
                int neuronsInPreviousLayer = h[i - 1];
                for (int j = 0; j < h[i]; j++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];
                    for (int k = 0; k < neuronsInPreviousLayer; k++)
                    {
                            neuronWeights[k] = Utils.RandomRange(-0.5f, 0.5f);
                    }
                    layerWeightsList.Add(neuronWeights);
                }
                weightsList.Add(layerWeightsList.ToArray());
            }
            w = weightsList.ToArray();
        }

        public void Train(List<(float[], float[])> _Sn, int _epochs = 20000)
        {
            for(int i = 1; i<= _epochs;i++)
            {
                for(int j = 0; j<_Sn.Count;j++)
                {
                    output.Add(Learn(_Sn[j].Item1, _Sn[j].Item2, n));
                    n++;
                }
            }
        }

        public float Learn(float[] _x, float[] _d, int _n)
        {
            float[] y = Run(_x);
            float e = Utils.Error(y,_d);
            float[][] delta;


            List<float[]> deltaList = new List<float[]>();
            for (int i = 0; i < h.Length; i++)
            {
                deltaList.Add(new float[h[i]]);
            }
            delta = deltaList.ToArray();//inicjalizacja delt

            int layer = h.Length - 2;
            for (int i = 0; i < y.Length; i++) delta[h.Length - 1][i] = -(_d[i] - y[i]) * beta * Utils.dSigmoid(y[i], beta) * GetGamma(_n);//Gamma calculation
            for (int i = 0; i < h[h.Length - 1]; i++)//obliczanie w' dla ostatniej warstwy
            {
                for (int j = 0; j < h[h.Length - 2]; j++)
                {
                    w[h.Length - 2][i][j] -= delta[h.Length - 1][i] * v[h.Length - 2][j];//nauka
                }
            }

            for (int i = h.Length - 2; i > 0; i--)//dla wszystkich warstw ukrytych
            {
                layer = i - 1;
                for (int j = 0; j < h[i]; j++)//wyliczanie wyjść
                {
                    delta[i][j] = 0;
                    for (int k = 0; k < delta[i + 1].Length; k++)
                    {
                        delta[i][j] += delta[i + 1][k] * w[i][k][j];
                    }
                    delta[i][j] *= Utils.dSigmoid(v[i][j], beta);//oblicznie delt
                }
                for (int j = 0; j < h[i]; j++)//iterowanie po wyjściach
                {
                    for (int k = 0; k < h[i - 1]; k++)//iterowanie po wejściach
                    {
                        w[i - 1][j][k] -= delta[i][j] * v[i - 1][k] * GetGamma(_n);//modyfikacja wag neuronu
                    }
                }
            }

            return e;
        }

        public float[] Run(float[] _x)
        {
            for (int i = 0; i < _x.Length; i++)
            {
               v[0][i] = _x[i];
            }
            for (int i = 1; i < h.Length; i++)
            {
                for (int j = 0; j < h[i]; j++)
                {
                    float value = 0f;
                    for (int k = 0; k < h[i - 1]; k++)
                    {
                        value += w[i - 1][j][k] * v[i - 1][k];
                    }
                    v[i][j] = Utils.Sigmoid(value, beta);
                }
            }
            return v[h.Length - 1];
        }


    }

    public enum ActivationFunctions
    {
        Sigmoid = 0
    }
}
