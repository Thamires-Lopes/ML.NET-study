using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NET_study.Entities
{
    public class BeanData
    {
        [LoadColumn(0)]
        public float Area;

        [LoadColumn(1)]
        public float Perimeter;

        [LoadColumn(2)]
        public float MajorAxisLength;

        [LoadColumn(3)]
        public float MinorAxisLength;

        [LoadColumn(4)]
        public float AspectRation;

        [LoadColumn(5)]
        public float Eccentricity;

        [LoadColumn(6)]
        public float ConvexArea;

        [LoadColumn(7)]
        public float EquivDiameter;

        [LoadColumn(8)]
        public float Extent;

        [LoadColumn(9)]
        public float Solidity;

        [LoadColumn(10)]
        public float Roundness;

        [LoadColumn(11)]
        public float Compactness;

        [LoadColumn(12)]
        public float ShapeFactor1;

        [LoadColumn(13)]
        public float ShapeFactor2;

        [LoadColumn(14)]
        public float ShapeFactor3;

        [LoadColumn(15)]
        public float ShapeFactor4;

        //[LoadColumn(16)]
        //public float Class;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[]? Distances;
    }
}
