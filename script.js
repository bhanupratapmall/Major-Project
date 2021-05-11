function calculate()
{
    ba=document.getElementById("BisphenolA").value;
    dgeba=document.getElementById("dgeba").value;
    kat=document.getElementById("kat").value;
    time=document.getElementById("time").value;
    temp=document.getElementById("temperatur").value;
    wpe=474.204+76.762*ba-76.027*dgeba+7.824*kat+6.585*temp+8.242*time;
    document.getElementById("result").innerHTML=wpe;
}