#!/usr/bin/python
# -*- coding: utf-8 -*-

atoms = {
    'H': {
        'a1': 0.493002,
        'b1': 10.5109,
        'a2': 0.322912,
        'b2': 26.1257,
        'a3': 0.140191,
        'b3': 3.14236,
        'a4': 0.04081,
        'b4': 57.7997,
        'c': 0.003038
    },
    'He': {
        'a1': 0.873400,
        'b1': 9.10370,
        'a2': 0.630900,
        'b2': 3.35680,
        'a3': 0.311200,
        'b3': 22.9276,
        'a4': 0.178000,
        'b4': 0.982100,
        'c': 0.006400
    },
    'Li': {
        'a1': 1.1282,
        'b1': 3.9546,
        'a2': 0.7508,
        'b2': 1.0524,
        'a3': 0.6175,
        'b3': 85.3905,
        'a4': 0.4653,
        'b4': 168.261,
        'c': 0.0377
    },
    'B': {
        'a1': 2.0545,
        'b1': 23.2185,
        'a2': 1.3326,
        'b2': 1.021,
        'a3': 1.0979,
        'b3': 60.3498,
        'a4': 0.7068,
        'b4': 0.1403,
        'c': -0.1932
        },
     'C': {
        'a1': 2.31000,
        'b1': 20.8439,
        'a2': 1.02000,
        'b2': 10.2075,
        'a3': 1.58860,
        'b3': 0.568700,
        'a4': 0.865000,
        'b4': 51.6512,
        'c': 0.215600
    },
    'N': {
        'a1': 12.2126,
        'b1': 0.005700,
        'a2': 3.13220,
        'b2': 9.89330,
        'a3': 2.01250,
        'b3': 28.9975,
        'a4': 1.16630,
        'b4': 0.582600,
        'c': -11.529
    },
    'O': {
        'a1': 3.04850,
        'b1': 13.2771,
        'a2': 2.28680,
        'b2': 5.70110,
        'a3': 1.54630,
        'b3': 0.323900,
        'a4': 0.867000,
        'b4': 32.9089,
        'c': 0.250800
    },
    'F': {
        'a1': 3.5392,
        'b1': 10.2825,
        'a2': 2.6412,
        'b2': 4.2944,
        'a3': 1.517,
        'b3': 0.2615,
        'a4': 1.0243,
        'b4': 26.1476,
        'c': 0.2776
    },
    'Na': {
        'a1': 4.76260,
        'b1': 3.28500,
        'a2': 3.17360,
        'b2': 8.8422,
        'a3': 1.26740,
        'b3': 0.313600,
        'a4': 1.11280,
        'b4': 129.424,
        'c': 0.676000
    },
    'Mg': {
        'a1': 5.42040,
        'b1': 2.82750,
        'a2': 2.17350,
        'b2': 79.2611,
        'a3': 1.22690,
        'b3': 0.380800,
        'a4': 2.30730,
        'b4': 7.19370,
        'c': 0.858400
    },
    'Al': {
        'a1': 6.42020,
        'b1': 3.03870,
        'a2': 1.90020,
        'b2': 0.742600,
        'a3': 1.59360,
        'b3': 31.5472,
        'a4': 1.96460,
        'b4': 85.0886,
        'c': 1.11510
    },
    'Si': {
        'a1': 6.2915,
        'b1': 2.4386,
        'a2': 3.0353,
        'b2': 32.3337,
        'a3': 1.9891,
        'b3': 0.6785,
        'a4': 1.541,
        'b4': 81.6937,
        'c': 1.1407
    },
    'P': {
        'a1': 6.43450,
        'b1': 1.90670,
        'a2': 4.17910,
        'b2': 27.1570,
        'a3': 1.78000,
        'b3': 0.52600,
        'a4': 1.49080,
        'b4': 68.1645,
        'c': 1.11490
    },
    'S': {
        'a1': 6.9053,
        'b1': 1.4679,
        'a2': 5.2034,
        'b2': 22.2151,
        'a3': 1.4379,
        'b3': 0.2536,
        'a4': 1.5863,
        'b4': 56.172,
        'c': 0.8669
    },
    'Cl': {
        'a1': 11.4604,
        'b1': 0.01040,
        'a2': 7.19640,
        'b2': 1.16620,
        'a3': 6.25560,
        'b3': 18.5194,
        'a4': 1.64550,
        'b4': 47.7784,
        'c': -9.5574
    },
    'K': {
        'a1': 8.21860,
        'b1': 12.7949,
        'a2': 7.43980,
        'b2': 0.77480,
        'a3': 1.05190,
        'b3': 213.187,
        'a4': 0.86590,
        'b4': 41.6841,
        'c': 1.42280
    },
    'Ca': {
        'a1': 8.62660,
        'b1': 10.4421,
        'a2': 7.38730,
        'b2': 0.65990,
        'a3': 1.58990,
        'b3': 85.7484,
        'a4': 1.02110,
        'b4': 178.437,
        'c': 1.37510
    },
    'Ti': {
        'a1': 9.7595,
        'b1': 7.8508,
        'a2': 7.3558,
        'b2': 0.5,
        'a3': 1.6991,
        'b3': 35.6338,
        'a4': 1.9021,
        'b4': 116.105,
        'c':  1.2807
    },
    'Mn': {
        'a1': 11.2819,
        'b1': 5.3409,
        'a2': 7.3573,
        'b2': 0.3432,
        'a3': 3.0193,
        'b3': 17.8674,
        'a4': 2.2441,
        'b4': 83.7543,
        'c': 1.0896
    },
    'Fe': {
        'a1': 11.7695,
        'b1': 4.7611,
        'a2': 7.3573,
        'b2': 0.3072,
        'a3': 3.5222,
        'b3': 15.3535,
        'a4': 2.3045,
        'b4': 76.8805,
        'c': 1.0369
    },
    'Co': {
        'a1': 12.2841,
        'b1': 4.2791,
        'a2': 7.3409,
        'b2': 0.2784,
        'a3': 4.0034,
        'b3': 13.5359,
        'a4': 2.3488,
        'b4': 71.1692,
        'c': 1.0118
    },
    'Cu': {
        'a1': 13.338,
        'b1': 3.5828,
        'a2': 7.1676,
        'b2': 0.247,
        'a3': 5.6158,
        'b3': 11.3966,
        'a4': 1.6735,
        'b4': 64.8126,
        'c': 1.191
    },
    'Ga': {
        'a1': 15.2354,
        'b1': 3.0669,
        'a2': 6.7006,
        'b2': 0.2412,
        'a3': 4.3591,
        'b3': 10.7805,
        'a4': 2.9623,
        'b4': 61.4135,
        'c':  1.7189
    },
    'Br':{
        'a1': 17.1789,
        'b1': 2.1723,
        'a2': 5.2358,
        'b2': 16.5796,
        'a3': 5.6377,
        'b3': 0.2609,
        'a4': 3.9851,
        'b4': 41.4328,
        'c': 2.9557
    },
    'Y':{
        'a1': 17.776,
        'b1': 1.4029,
        'a2': 10.2946,
        'b2': 12.8006,
        'a3': 5.72629,
        'b3': 0.125599,
        'a4': 3.26588,
        'b4': 104.354,
        'c': 1.91213
    },
    'Ce': {
        'a1': 21.1671,
        'b1': 2.81219,
        'a2': 19.7695,
        'b2': 0.226836,
        'a3': 11.8513,
        'b3': 17.6083,
        'a4': 3.33049,
        'b4': 127.113,
        'c': 1.86264
    },
}
