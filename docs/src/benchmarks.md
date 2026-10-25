# Benchmarks

Measured on an AVX2 laptop (Intel Core Ultra 7 268V, 4 performance and 4
efficiency cores) unless noted; fresh input every round, methods interleaved
in random order, best of the rounds. The tables with all cases are in the
README; the scripts are in `bench/` (`bench.jl`, `types.jl`, `crossover.jl`,
`tables.jl`, `plots.jl`).

## Primitive types

Seconds per sort; 2^26 keys (2^27 for `UInt64`). In every row the fastest
result with one thread and with 8 threads is highlighted (bold); results
within 10% of it are lightly tinted.

<!-- table:main -->
```@raw html
<div style="overflow-x: auto;">
<table>
  <thead>
    <tr class = "columnLabelRow">
      <th style = "font-weight: bold; text-align: left;">case</th>
      <th style = "font-weight: bold; text-align: right;">Base 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">x86-simd-sort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">vqsort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 8 thr</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 8 thr</th>
      <th style = "font-weight: bold; text-align: right;">x86-simd-sort 8 thr</th>
    </tr>
  </thead>
  <tbody>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64, 2^27</td>
      <td style = "text-align: right;">3.084</td>
      <td style = "text-align: right;">1.354</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.140</td>
      <td style = "text-align: right;">1.827</td>
      <td style = "text-align: right;">1.546</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.305</td>
      <td style = "text-align: right;">0.399</td>
      <td style = "text-align: right;">0.531</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int64, 2^26</td>
      <td style = "text-align: right;">1.495</td>
      <td style = "text-align: right;">0.800</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.497</td>
      <td style = "text-align: right;">0.742</td>
      <td style = "text-align: right;">0.696</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.175</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.166</td>
      <td style = "text-align: right;">0.232</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 (randn), 2^26</td>
      <td style = "text-align: right;">1.519</td>
      <td style = "text-align: right;">0.954</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.508</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.536</td>
      <td style = "text-align: right;">0.735</td>
      <td style = "text-align: right;">0.197</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.172</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.173</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt32, 2^26</td>
      <td style = "text-align: right;">0.481</td>
      <td style = "text-align: right;">0.438</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.256</td>
      <td style = "text-align: right;">0.300</td>
      <td style = "text-align: right;">0.282</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.089</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.082</td>
      <td style = "text-align: right;">0.092</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32, 2^26</td>
      <td style = "text-align: right;">0.480</td>
      <td style = "text-align: right;">0.469</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.258</td>
      <td style = "text-align: right;">0.312</td>
      <td style = "text-align: right;">0.290</td>
      <td style = "text-align: right;">0.102</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.084</td>
      <td style = "text-align: right;">0.094</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float32 (randn), 2^26</td>
      <td style = "text-align: right;">0.663</td>
      <td style = "text-align: right;">0.490</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.280</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.307</td>
      <td style = "text-align: right;">0.310</td>
      <td style = "text-align: right;">0.107</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.089</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.095</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 in 0:1000, 2^26</td>
      <td style = "text-align: right;">0.077</td>
      <td style = "text-align: right;">0.081</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.061</td>
      <td style = "text-align: right;">0.305</td>
      <td style = "text-align: right;">0.255</td>
      <td style = "text-align: right;">0.033</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.020</td>
      <td style = "text-align: right;">0.131</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64, 1000 random values, 2^26</td>
      <td style = "text-align: right;">1.548</td>
      <td style = "text-align: right;">0.926</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.111</td>
      <td style = "text-align: right;">0.314</td>
      <td style = "text-align: right;">0.267</td>
      <td style = "text-align: right;">0.123</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.041</td>
      <td style = "text-align: right;">0.134</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 sorted, 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.027</td>
      <td style = "text-align: right;">0.028</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.025</td>
      <td style = "text-align: right;">0.913</td>
      <td style = "text-align: right;">0.684</td>
      <td style = "text-align: right;">0.009</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.008</td>
      <td style = "text-align: right;">0.266</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 reversed, 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.052</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.054</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.051</td>
      <td style = "text-align: right;">0.914</td>
      <td style = "text-align: right;">0.660</td>
      <td style = "text-align: right;">0.021</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.019</td>
      <td style = "text-align: right;">0.256</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32, 2^20</td>
      <td style = "text-align: right;">0.005</td>
      <td style = "text-align: right;">0.007</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.003</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.003</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.003</td>
      <td style = "text-align: right;">0.002</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.001</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.002</td>
    </tr>
  </tbody>
</table>
</div>
```
<!-- /table:main -->

![Random keys, one thread](assets/random1.svg)

![Random keys, 8 threads](assets/random8.svg)

![Duplicates and presorted inputs](assets/special.svg)

## Other types

10^7 elements (BigInt, BigFloat, Float64x4, VersionNumber: 10^6), best of 3
rounds.

<!-- table:types -->
```@raw html
<div style="overflow-x: auto;">
<table>
  <thead>
    <tr class = "columnLabelRow">
      <th style = "font-weight: bold; text-align: left;">case, operation</th>
      <th style = "font-weight: bold; text-align: right;">Base 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 8 thr</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 8 thr</th>
    </tr>
  </thead>
  <tbody>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, random, length 8–20, sort</td>
      <td style = "text-align: right;">4.916</td>
      <td style = "text-align: right;">1.373</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.707</td>
      <td style = "text-align: right;">0.304</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.235</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, random, length 8–20, sortperm</td>
      <td style = "text-align: right;">11.578</td>
      <td style = "text-align: right;">1.739</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.618</td>
      <td style = "text-align: right;">0.508</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.159</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, h2o id3 (id%010d), sort</td>
      <td style = "text-align: right;">3.199</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.159</td>
      <td style = "text-align: right;">1.749</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.367</td>
      <td style = "text-align: right;">0.495</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, h2o id3 (id%010d), sortperm</td>
      <td style = "text-align: right;">7.138</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.512</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">1.587</td>
      <td style = "text-align: right;">0.672</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.365</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol, 10^5 distinct, sort</td>
      <td style = "text-align: right;">1.082</td>
      <td style = "text-align: right;">0.298</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.042</td>
      <td style = "text-align: right;">0.096</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.029</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol, 10^5 distinct, sortperm</td>
      <td style = "text-align: right;">3.301</td>
      <td style = "text-align: right;">0.468</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.348</td>
      <td style = "text-align: right;">0.217</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.117</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128, sort</td>
      <td style = "text-align: right;">0.410</td>
      <td style = "text-align: right;">0.273</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.187</td>
      <td style = "text-align: right;">0.073</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.056</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128, sortperm</td>
      <td style = "text-align: right;">1.688</td>
      <td style = "text-align: right;">0.426</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.227</td>
      <td style = "text-align: right;">0.103</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.071</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt256 (BitIntegers), sort</td>
      <td style = "text-align: right;">0.655</td>
      <td style = "text-align: right;">0.345</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.272</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.085</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.092</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt256 (BitIntegers), sortperm</td>
      <td style = "text-align: right;">2.853</td>
      <td style = "text-align: right;">0.604</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.231</td>
      <td style = "text-align: right;">0.176</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.079</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Double64 (DoubleFloats), sort</td>
      <td style = "text-align: right;">0.930</td>
      <td style = "text-align: right;">0.705</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.249</td>
      <td style = "text-align: right;">0.163</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.068</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Double64 (DoubleFloats), sortperm</td>
      <td style = "text-align: right;">2.346</td>
      <td style = "text-align: right;">0.520</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.366</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.141</td>
      <td style = "text-align: right;">0.200</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float128 (Quadmath), sort</td>
      <td style = "text-align: right;">2.590</td>
      <td style = "text-align: right;">0.390</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.214</td>
      <td style = "text-align: right;">0.087</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.067</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float128 (Quadmath), sortperm</td>
      <td style = "text-align: right;">7.756</td>
      <td style = "text-align: right;">0.502</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.345</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.121</td>
      <td style = "text-align: right;">0.173</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x2 (MultiFloats), sort</td>
      <td style = "text-align: right;">0.808</td>
      <td style = "text-align: right;">0.718</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.266</td>
      <td style = "text-align: right;">0.158</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.065</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x2 (MultiFloats), sortperm</td>
      <td style = "text-align: right;">3.933</td>
      <td style = "text-align: right;">0.461</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.287</td>
      <td style = "text-align: right;">0.141</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.117</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x4 (MultiFloats), sort</td>
      <td style = "text-align: right;">0.069</td>
      <td style = "text-align: right;">0.047</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.028</td>
      <td style = "text-align: right;">0.011</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.008</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x4 (MultiFloats), sortperm</td>
      <td style = "text-align: right;">0.296</td>
      <td style = "text-align: right;">0.057</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.017</td>
      <td style = "text-align: right;">0.015</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.008</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Date, sort</td>
      <td style = "text-align: right;">0.168</td>
      <td style = "text-align: right;">0.086</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.051</td>
      <td style = "text-align: right;">0.024</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.021</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Date, sortperm</td>
      <td style = "text-align: right;">1.528</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.139</td>
      <td style = "text-align: right;">0.253</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.037</td>
      <td style = "text-align: right;">0.085</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">DateTime, sort</td>
      <td style = "text-align: right;">0.227</td>
      <td style = "text-align: right;">0.208</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.067</td>
      <td style = "text-align: right;">0.036</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.023</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">DateTime, sortperm</td>
      <td style = "text-align: right;">1.556</td>
      <td style = "text-align: right;">0.324</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.251</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.079</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.082</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,Float64}, sort</td>
      <td style = "text-align: right;">0.707</td>
      <td style = "text-align: right;">0.497</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.217</td>
      <td style = "text-align: right;">0.113</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.060</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,Float64}, sortperm</td>
      <td style = "text-align: right;">2.298</td>
      <td style = "text-align: right;">0.560</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.349</td>
      <td style = "text-align: right;">0.140</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.109</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Pair{Int32,Int64}, sort</td>
      <td style = "text-align: right;">0.441</td>
      <td style = "text-align: right;">0.254</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.196</td>
      <td style = "text-align: right;">0.069</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.060</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Pair{Int32,Int64}, sortperm</td>
      <td style = "text-align: right;">1.969</td>
      <td style = "text-align: right;">0.445</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.293</td>
      <td style = "text-align: right;">0.112</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.094</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,String}, sort</td>
      <td style = "text-align: right;">0.633</td>
      <td style = "text-align: right;">0.390</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.299</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.164</td>
      <td style = "text-align: right;">0.193</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,String}, sortperm</td>
      <td style = "text-align: right;">1.409</td>
      <td style = "text-align: right;">0.355</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.282</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.133</td>
      <td style = "text-align: right;">0.171</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">VersionNumber, sort</td>
      <td style = "text-align: right;">0.140</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.061</td>
      <td style = "text-align: right;">0.071</td>
      <td style = "text-align: right;">0.043</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.039</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">VersionNumber, sortperm</td>
      <td style = "text-align: right;">0.261</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.045</td>
      <td style = "text-align: right;">0.058</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.028</td>
      <td style = "text-align: right;">0.033</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigInt (~128-384 bits), sort</td>
      <td style = "text-align: right;">0.431</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.139</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.137</td>
      <td style = "text-align: right;">0.081</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.052</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigInt (~128-384 bits), sortperm</td>
      <td style = "text-align: right;">0.901</td>
      <td style = "text-align: right;">0.183</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.166</td>
      <td style = "text-align: right;">0.092</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.052</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigFloat (256 bits), sort</td>
      <td style = "text-align: right;">0.495</td>
      <td style = "text-align: right;">0.213</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.128</td>
      <td style = "text-align: right;">0.085</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.054</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigFloat (256 bits), sortperm</td>
      <td style = "text-align: right;">0.858</td>
      <td style = "text-align: right;">0.204</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.118</td>
      <td style = "text-align: right;">0.089</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.055</td>
    </tr>
  </tbody>
</table>
</div>
```
<!-- /table:types -->

![sortperm speedup over Base](assets/types_sortperm.svg)

![sort speedup over Base](assets/types_sort.svg)

## Quicksort or radix sort?

![Crossover between ChapelSort and SIMDSort](assets/crossover.svg)

## Many cores and AVX-512

Thread scaling on the laptop (AVX2, 8 threads) and on a 2 × 28-core Xeon
Gold 6348 (AVX-512, 56 cores), same commit; dashed: SIMDSort with only one of
its two top levels (quicksort, or the sample-sort step forced on):

![Thread scaling on AVX2 and AVX-512](assets/scaling.svg)

The Xeon at one and 56 threads (seconds; `bench/measure/scaling.jl`, with the
512-bit kernels):

<!-- table:xeon -->
```@raw html
<div style="overflow-x: auto;">
<table>
  <thead>
    <tr class = "columnLabelRow">
      <th style = "font-weight: bold; text-align: left;">case</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">x86-simd-sort 1 thr</th>
      <th style = "font-weight: bold; text-align: right;">SIMDSort 56 thr</th>
      <th style = "font-weight: bold; text-align: right;">ChapelSort 56 thr</th>
      <th style = "font-weight: bold; text-align: right;">x86-simd-sort 56 thr</th>
    </tr>
  </thead>
  <tbody>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 2^27</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.445</td>
      <td style = "text-align: right;">3.005</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">1.485</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.102</td>
      <td style = "text-align: right;">0.116</td>
      <td style = "text-align: right;">0.275</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int64 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.704</td>
      <td style = "text-align: right;">1.696</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.738</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.045</td>
      <td style = "text-align: right;">0.083</td>
      <td style = "text-align: right;">0.133</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 randn 2^26</td>
      <td style = "text-align: right;">0.753</td>
      <td style = "text-align: right;">1.815</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.674</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.060</td>
      <td style = "text-align: right;">0.097</td>
      <td style = "text-align: right;">0.127</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.357</td>
      <td style = "text-align: right;">1.094</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.347</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.034</td>
      <td style = "text-align: right;">0.047</td>
      <td style = "text-align: right;">0.063</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.270</td>
      <td style = "text-align: right;">0.445</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.015</td>
      <td style = "text-align: right;">0.021</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 sortperm 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.296</td>
      <td style = "text-align: right;">0.499</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.021</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.022</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String 8-20 sortperm 2*10^6</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.102</td>
      <td style = "text-align: right;">0.232</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.012</td>
      <td style = "text-align: right;">0.021</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol 10^5 distinct sort 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.073</td>
      <td style = "text-align: right;">0.403</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.017</td>
      <td style = "text-align: right;">0.028</td>
      <td style = "text-align: right;">–</td>
    </tr>
  </tbody>
</table>
</div>
```
<!-- /table:xeon -->

More results on the Xeon are summarized in
[AVX-512](avx512.md); the design document's performance analysis
([Design](design.md), §7) has the per-level costs and the effect of each
optimization.
