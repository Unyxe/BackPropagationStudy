using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation
{
    public class Neuron
    {
        public string id = "Neuron";
        Random rand;
        public double[] weights;
        public double bias;
        public double[] last_input;
        public double last_output;
        public double last_error;
        public Neuron(int num_of_weights, Random rn_cl)
        {
            rand = rn_cl;
            weights = new double[num_of_weights];
            for(int i = 0; i < num_of_weights; i++)
            {
                weights[i] = rand.NextDouble() * 2 -1;
            }
            bias = rand.NextDouble() * 2 - 1;
        }
        public Neuron(double[] weights_, double bias_, Random rn_cl)
        {
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] = weights_[i];
            }
            bias = bias_;
        }
        public void print_weights()
        {
            foreach(double w in weights)
            {
                Console.WriteLine(w);
            }
        }
        public void adjustWeights()
        {
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] += last_error * last_input[i];
            }
            bias += last_error;
        }
        public double calculate_output_from_input(double[] input)
        {
            last_input = input;
            double output_sum = 0;
            for(int i = 0; i < input.Length; i++)
            {
                output_sum += input[i] * weights[i];
            }
            double res = NeuralNetwork.Activation(output_sum + bias);
            last_output = res;
            //Console.WriteLine(output_sum+"    "+ res);
            return res;
        }
        public double[] CloneWeights()
        {
            Random rn = new Random();
            double[] m_weights = new double[weights.Length];
            for(int i = 0; i < weights.Length; i++)
            {
                m_weights[i] = weights[i];
            }
            //m_weights[rn.Next() % m_weights.Length] = rn.NextDouble();
            return m_weights;
        }
        public void SetWeights(double[] weig)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = weig[i];
            }
        }
    }
    public class NeuralNetwork
    {
        Random rand;
        public List<List<Neuron>> NN = new List<List<Neuron>>();
        public int num_outputs;
        public int num_input;
        bool debug_on = false;
        
        public NeuralNetwork(int hidden_layers, int neurons_per_hidden_layer, int num_of_inputs, int num_of_outputs, int seed, bool debug_info)
        {
            rand = new Random(seed);
            num_outputs = num_of_outputs;
            num_input = num_of_inputs;
            debug_on = debug_info;
            if (debug_on) {
                Console.WriteLine("Initializating weights");
            }
            for(int i = 0; i < hidden_layers;i++)
            {
                List<Neuron> layer = new List<Neuron>();
                for(int j = 0; j < neurons_per_hidden_layer; j++)
                {
                    
                    if (i == 0)
                    {
                        layer.Add(new Neuron(num_of_inputs, rand));
                        if (debug_on)
                        {
                            Console.WriteLine(i + " " + j + " " + layer[j].weights.Length + layer[j].id);
                            layer[j].print_weights();
                            Console.WriteLine();
                        }
                        
                        continue;
                    }
                    layer.Add(new Neuron(neurons_per_hidden_layer, rand));
                    if (debug_on)
                    {
                        Console.WriteLine(i + " " + j + " " + layer[j].weights.Length + layer[j].id);
                        layer[j].print_weights();
                        Console.WriteLine();
                    }
                    
                }
                NN.Add(layer);
            }
            List<Neuron> output_layer = new List<Neuron>();
            for(int i = 0;i<num_of_outputs; i++)
            {
                output_layer.Add(new Neuron(neurons_per_hidden_layer, rand));
                if (debug_on)
                {
                    Console.WriteLine("out" + " " + i + " " + output_layer[i].weights.Length + output_layer[i].id);
                    output_layer[i].print_weights();
                    Console.WriteLine();
                }
            }
            NN.Add(output_layer);
        }
        public static double Activation(double i)
        {
            return 1.0 / (1.0 + Math.Exp(-i));
        }
        public static double ActivationDerivative(double i)
        {
            return i * (1 - i);
        }
        public void Train(double[][] dataset, int iter)
        {
            for(int i = 0; i < iter; i++)
            {
                double[] input = new double[num_input];
                double[] expected_output = new double[num_outputs];
                double[] output;
                int rand_ind = rand.Next() % dataset.Length;
                for(int j = 0; j < input.Length; j++)
                {
                    input[j] = dataset[rand_ind][j];
                }
                for (int j = 0; j < expected_output.Length; j++)
                {
                    expected_output[j] = dataset[rand_ind][j+num_input];
                }
                //Console.WriteLine(input.Length+"    "+ expected_output.Length);
                output = CalculateOutput(input);
                double global_error = 0;
                for(int j = 0; j < output.Length; j++)
                {
                    global_error += ActivationDerivative(output[j]) * (expected_output[j] - output[j]);
                }

                if (i % 10000 == 0)
                {
                    Console.WriteLine("Error: " + global_error);
                }

                List<Neuron> output_layer = NN[NN.Count - 1];
                for(int j = 0; j < output_layer.Count; j++)
                {
                    output_layer[j].last_error = global_error;
                    output_layer[j].adjustWeights();
                }

                for(int j = NN.Count - 2; j >=0; j--)
                {
                    List<Neuron> layer = NN[j];
                    for(int k = 0; k < layer.Count; k++)
                    {
                        double r = layer[k].last_output;
                        double s = NN[j + 1][0].weights[k];
                        layer[k].last_error = ActivationDerivative(r * global_error * s);
                        layer[k].adjustWeights();
                    }
                }
            }
        }

        public double[] CalculateOutput(double[] input)
        {
            if (debug_on)
            {
                Console.WriteLine("Calculating output");
            }
            double[] next_input;
            double[] current_input = new double[input.Length];
            for(int i =0; i < input.Length; i++)
            {
                current_input[i] = input[i];
            }
            for(int i = 0; i < NN.Count; i++)
            {
                next_input = new double[NN[i].Count];
                for(int j = 0; j < NN[i].Count; j++)
                {
                    if (debug_on)
                    {
                        Console.WriteLine(i + " " + j);
                    }
                    next_input[j] = NN[i][j].calculate_output_from_input(current_input);

                }
                current_input = new double[next_input.Length];
                for(int j = 0; j < next_input.Length; j++)
                {
                    current_input[j] = next_input[j];
                }
            }
            return current_input;
        }


        public List<double[]> ExportWeights()
        {
            List<double[]> exp_weights = new List<double[]>();
            foreach(List<Neuron> layer in NN)
            {
                foreach(Neuron n in layer)
                {
                    exp_weights.Add(n.CloneWeights());
                }
            }
            return exp_weights;
        }

        public void SetWeights(List<double[]> weights_e)
        {
            int ind = 0;
            foreach (List<Neuron> layer in NN)
            {
                foreach (Neuron n in layer)
                {
                    n.SetWeights(weights_e[ind]);
                    ind++;
                }
            }
        }
        public List<double[]> MiosisWeights(List<double[]> weights_m)
        {
            Random rn = new Random();
            List<double[]> exp_weights = this.ExportWeights();
            for(int i = 0; i < exp_weights.Count; i++)
            {
                for(int j = 0; j< exp_weights[i].Length; j++)
                {
                    if (rn.Next() % 2 == 0)
                    {
                        exp_weights[i][j] = weights_m[i][j];
                    }
                }
            }
            return exp_weights;
        }
    }
}
