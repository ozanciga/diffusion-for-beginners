## diffusion for beginners

- implementation of _diffusion schedulers_ (currently sampling only) with minimal code & as faithful to the original work as i could. most recent work reuse or adopt code from previous work and build on it, or transcribe code from another framework - which is great! but i found it hard to follow at times. this is an attempt at simplifying below great papers. the trade-off is made between stability and correctness vs. brevity and simplicity. 

- [x] ddpm (ho et al. 2020), https://arxiv.org/abs/2006.11239
- [x] improved ddpm (nichol and dhariwal 2021), https://arxiv.org/abs/2102.09672
- [ ] ddim (song et al. 2020), https://arxiv.org/abs/2010.02502
- [ ] pndm (ho et al. 2020), https://arxiv.org/abs/2202.09778
- [ ] heun (karras et al. 2020), https://arxiv.org/abs/2206.00364


**prompt**: "a man eating an apple sitting on a bench"

<table>
 <tr>
    <td><img src="images/ddpm.jpg" height="256" width="256"></td>
    <td><img src='images/improved_ddpm.jpg' height="256" width="256"></td>
 </tr>
 <tr>
   <td><b style="font-size:20px">ddpm</b></td>
   <td><b style="font-size:20px">improved ddpm</b></td>
 </tr>
</table>


<table>
 <tr>
    <td><img src='images/ddim.jpg' height="256" width="256"></td>
    <td><img src='data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==' height="256" width="256"></td>
 </tr>
 <tr>
   <td><b style="font-size:20px">ddim (wip)</b></td>
   <td><b style="font-size:20px">** </b></td>
 </tr>
</table>


<table>
 <tr>
    <td><img src='data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==' height="256" width="256"></td>
    <td><img src='data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==' height="256" width="256"></td>
 </tr>
 <tr>
   <td><b style="font-size:20px">pndm (wip)</b></td>
   <td><b style="font-size:20px">heun (wip)</b></td>
 </tr>
</table>

