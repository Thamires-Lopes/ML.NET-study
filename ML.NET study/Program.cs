using BenchmarkDotNet.Running;
using ML.NET_study.Algorithms;

namespace ML.NET_study
{
    public class Program
    {
        static void Main()
        {
            var results = BenchmarkRunner.Run<KMeans>();
        }
    }
}