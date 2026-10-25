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
      <td style = "text-align: right;">3.062</td>
      <td style = "text-align: right;">1.362</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.168</td>
      <td style = "text-align: right;">1.818</td>
      <td style = "text-align: right;">1.575</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.300</td>
      <td style = "text-align: right;">0.406</td>
      <td style = "text-align: right;">0.542</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int64, 2^26</td>
      <td style = "text-align: right;">1.513</td>
      <td style = "text-align: right;">0.787</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.501</td>
      <td style = "text-align: right;">0.752</td>
      <td style = "text-align: right;">0.696</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.176</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.168</td>
      <td style = "text-align: right;">0.232</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 (randn), 2^26</td>
      <td style = "text-align: right;">1.453</td>
      <td style = "text-align: right;">0.947</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.512</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.538</td>
      <td style = "text-align: right;">0.736</td>
      <td style = "text-align: right;">0.202</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.172</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.172</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt32, 2^26</td>
      <td style = "text-align: right;">0.484</td>
      <td style = "text-align: right;">0.435</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.251</td>
      <td style = "text-align: right;">0.299</td>
      <td style = "text-align: right;">0.281</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.087</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.082</td>
      <td style = "text-align: right;">0.092</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32, 2^26</td>
      <td style = "text-align: right;">0.507</td>
      <td style = "text-align: right;">0.460</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.256</td>
      <td style = "text-align: right;">0.311</td>
      <td style = "text-align: right;">0.286</td>
      <td style = "text-align: right;">0.106</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.084</td>
      <td style = "text-align: right;">0.096</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float32 (randn), 2^26</td>
      <td style = "text-align: right;">0.668</td>
      <td style = "text-align: right;">0.483</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.280</td>
      <td style = "text-align: right;">0.310</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.307</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.102</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.094</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.094</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 in 0:1000, 2^26</td>
      <td style = "text-align: right;">0.079</td>
      <td style = "text-align: right;">0.081</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.061</td>
      <td style = "text-align: right;">0.307</td>
      <td style = "text-align: right;">0.259</td>
      <td style = "text-align: right;">0.028</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.025</td>
      <td style = "text-align: right;">0.128</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64, 1000 random values, 2^26</td>
      <td style = "text-align: right;">1.519</td>
      <td style = "text-align: right;">0.927</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.119</td>
      <td style = "text-align: right;">0.309</td>
      <td style = "text-align: right;">0.266</td>
      <td style = "text-align: right;">0.122</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.037</td>
      <td style = "text-align: right;">0.132</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 sorted, 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.028</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.028</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.026</td>
      <td style = "text-align: right;">0.917</td>
      <td style = "text-align: right;">0.677</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.009</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.009</td>
      <td style = "text-align: right;">0.272</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt64 reversed, 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.052</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.054</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.050</td>
      <td style = "text-align: right;">0.906</td>
      <td style = "text-align: right;">0.669</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.019</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.019</td>
      <td style = "text-align: right;">0.262</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32, 2^20</td>
      <td style = "text-align: right;">0.005</td>
      <td style = "text-align: right;">0.007</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.003</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.003</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.003</td>
      <td style = "text-align: right;">0.002</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.002</td>
      <td style = "text-align: right;">0.003</td>
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
      <td style = "text-align: right;">4.890</td>
      <td style = "text-align: right;">1.353</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.664</td>
      <td style = "text-align: right;">0.309</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.219</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, random, length 8–20, sortperm</td>
      <td style = "text-align: right;">11.766</td>
      <td style = "text-align: right;">1.733</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.560</td>
      <td style = "text-align: right;">0.519</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.142</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, h2o id3 (id%010d), sort</td>
      <td style = "text-align: right;">3.233</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">1.153</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.135</td>
      <td style = "text-align: right;">0.378</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.323</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String, h2o id3 (id%010d), sortperm</td>
      <td style = "text-align: right;">7.385</td>
      <td style = "text-align: right;">1.489</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.026</td>
      <td style = "text-align: right;">0.546</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.255</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol, 10^5 distinct, sort</td>
      <td style = "text-align: right;">1.130</td>
      <td style = "text-align: right;">0.306</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.042</td>
      <td style = "text-align: right;">0.095</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.023</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol, 10^5 distinct, sortperm</td>
      <td style = "text-align: right;">3.299</td>
      <td style = "text-align: right;">0.480</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.248</td>
      <td style = "text-align: right;">0.221</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.093</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128, sort</td>
      <td style = "text-align: right;">0.405</td>
      <td style = "text-align: right;">0.274</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.182</td>
      <td style = "text-align: right;">0.077</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.056</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128, sortperm</td>
      <td style = "text-align: right;">1.671</td>
      <td style = "text-align: right;">0.434</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.158</td>
      <td style = "text-align: right;">0.110</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.054</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt256 (BitIntegers), sort</td>
      <td style = "text-align: right;">0.632</td>
      <td style = "text-align: right;">0.341</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.264</td>
      <td style = "text-align: right;">0.085</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.073</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">UInt256 (BitIntegers), sortperm</td>
      <td style = "text-align: right;">2.528</td>
      <td style = "text-align: right;">0.644</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.163</td>
      <td style = "text-align: right;">0.165</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.061</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Double64 (DoubleFloats), sort</td>
      <td style = "text-align: right;">0.950</td>
      <td style = "text-align: right;">0.709</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.270</td>
      <td style = "text-align: right;">0.160</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.067</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Double64 (DoubleFloats), sortperm</td>
      <td style = "text-align: right;">2.285</td>
      <td style = "text-align: right;">0.529</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.229</td>
      <td style = "text-align: right;">0.146</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.080</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float128 (Quadmath), sort</td>
      <td style = "text-align: right;">2.673</td>
      <td style = "text-align: right;">0.387</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.208</td>
      <td style = "text-align: right;">0.095</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.063</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float128 (Quadmath), sortperm</td>
      <td style = "text-align: right;">8.671</td>
      <td style = "text-align: right;">0.552</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.303</td>
      <td style = "text-align: right;">0.123</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.101</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x2 (MultiFloats), sort</td>
      <td style = "text-align: right;">0.830</td>
      <td style = "text-align: right;">0.711</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.257</td>
      <td style = "text-align: right;">0.162</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.065</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x2 (MultiFloats), sortperm</td>
      <td style = "text-align: right;">4.191</td>
      <td style = "text-align: right;">0.465</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.238</td>
      <td style = "text-align: right;">0.132</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.076</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x4 (MultiFloats), sort</td>
      <td style = "text-align: right;">0.072</td>
      <td style = "text-align: right;">0.047</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.029</td>
      <td style = "text-align: right;">0.014</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.008</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64x4 (MultiFloats), sortperm</td>
      <td style = "text-align: right;">0.322</td>
      <td style = "text-align: right;">0.059</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.015</td>
      <td style = "text-align: right;">0.015</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.007</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Date, sort</td>
      <td style = "text-align: right;">0.191</td>
      <td style = "text-align: right;">0.097</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.051</td>
      <td style = "text-align: right;">0.025</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.021</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Date, sortperm</td>
      <td style = "text-align: right;">1.456</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.159</td>
      <td style = "text-align: right;">0.179</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.041</td>
      <td style = "text-align: right;">0.060</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">DateTime, sort</td>
      <td style = "text-align: right;">0.261</td>
      <td style = "text-align: right;">0.220</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.067</td>
      <td style = "text-align: right;">0.045</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.024</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">DateTime, sortperm</td>
      <td style = "text-align: right;">1.382</td>
      <td style = "text-align: right;">0.353</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.176</td>
      <td style = "text-align: right;">0.080</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.062</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,Float64}, sort</td>
      <td style = "text-align: right;">0.706</td>
      <td style = "text-align: right;">0.499</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.218</td>
      <td style = "text-align: right;">0.118</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.057</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,Float64}, sortperm</td>
      <td style = "text-align: right;">2.296</td>
      <td style = "text-align: right;">0.588</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.278</td>
      <td style = "text-align: right;">0.143</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.089</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Pair{Int32,Int64}, sort</td>
      <td style = "text-align: right;">0.451</td>
      <td style = "text-align: right;">0.257</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.193</td>
      <td style = "text-align: right;">0.066</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.048</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Pair{Int32,Int64}, sortperm</td>
      <td style = "text-align: right;">2.021</td>
      <td style = "text-align: right;">0.472</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.237</td>
      <td style = "text-align: right;">0.117</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.079</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,String}, sort</td>
      <td style = "text-align: right;">0.708</td>
      <td style = "text-align: right;">0.526</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.281</td>
      <td style = "text-align: right;">0.183</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.090</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Tuple{Int64,String}, sortperm</td>
      <td style = "text-align: right;">1.482</td>
      <td style = "text-align: right;">0.424</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.256</td>
      <td style = "text-align: right;">0.127</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.079</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">VersionNumber, sort</td>
      <td style = "text-align: right;">0.154</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.064</td>
      <td style = "text-align: right;">0.075</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.042</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.041</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">VersionNumber, sortperm</td>
      <td style = "text-align: right;">0.285</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.047</td>
      <td style = "text-align: right;">0.060</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.029</td>
      <td style = "text-align: right;">0.034</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigInt (~128-384 bits), sort</td>
      <td style = "text-align: right;">0.495</td>
      <td style = "text-align: right;">0.194</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.167</td>
      <td style = "text-align: right;">0.097</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.061</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigInt (~128-384 bits), sortperm</td>
      <td style = "text-align: right;">1.040</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.180</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.176</td>
      <td style = "text-align: right;">0.105</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.058</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigFloat (256 bits), sort</td>
      <td style = "text-align: right;">0.600</td>
      <td style = "text-align: right;">0.224</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.151</td>
      <td style = "text-align: right;">0.092</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.060</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">BigFloat (256 bits), sortperm</td>
      <td style = "text-align: right;">0.915</td>
      <td style = "text-align: right;">0.194</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.107</td>
      <td style = "text-align: right;">0.088</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.048</td>
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
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">1.493</td>
      <td style = "text-align: right;">3.044</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">1.525</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.096</td>
      <td style = "text-align: right;">0.121</td>
      <td style = "text-align: right;">0.274</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int64 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.690</td>
      <td style = "text-align: right;">1.672</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.727</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.051</td>
      <td style = "text-align: right;">0.082</td>
      <td style = "text-align: right;">0.134</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 randn 2^26</td>
      <td style = "text-align: right;">0.759</td>
      <td style = "text-align: right;">1.819</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.685</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.056</td>
      <td style = "text-align: right;">0.094</td>
      <td style = "text-align: right;">0.128</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int32 2^26</td>
      <td style = "background-color: rgba(46, 160, 67, 0.09); text-align: right;">0.353</td>
      <td style = "text-align: right;">1.093</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.347</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.033</td>
      <td style = "text-align: right;">0.047</td>
      <td style = "text-align: right;">0.062</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Int128 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.270</td>
      <td style = "text-align: right;">0.440</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.013</td>
      <td style = "text-align: right;">0.021</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Float64 sortperm 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.236</td>
      <td style = "text-align: right;">0.494</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.016</td>
      <td style = "text-align: right;">0.024</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">String 8-20 sortperm 2*10^6</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.094</td>
      <td style = "text-align: right;">0.227</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.011</td>
      <td style = "text-align: right;">0.021</td>
      <td style = "text-align: right;">–</td>
    </tr>
    <tr class = "dataRow">
      <td style = "text-align: left;">Symbol 10^5 distinct sort 10^7</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.069</td>
      <td style = "text-align: right;">0.403</td>
      <td style = "text-align: right;">–</td>
      <td style = "background-color: rgba(46, 160, 67, 0.25); font-weight: bold; text-align: right;">0.016</td>
      <td style = "text-align: right;">0.027</td>
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
