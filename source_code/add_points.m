function add_points(tensor_slice,bias,text_bias)
fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
center = floor(size(tensor_slice,1)/2);
plot(x_span(0+center),y_span(bias+center),'r*')
text(x_span(0+center)+text_bias,y_span(bias+center)-text_bias,num2str(tensor_slice(0+center,bias+center),'%.3f'))
hold on
plot(x_span(bias+center),y_span(0+center),'r*')
text(x_span(bias+center)+text_bias,y_span(0+center)-text_bias,num2str(tensor_slice(bias+center,0+center),'%.3f'))
hold on
plot(x_span(0+center),y_span(-bias+center),'r*')
text(x_span(0+center)+text_bias,y_span(-bias+center)-text_bias,num2str(tensor_slice(0+center,-bias+center),'%.3f'))
hold on
plot(x_span(-bias+center),y_span(0+center),'r*')
text(x_span(-bias+center)+text_bias,y_span(0+center)-text_bias,num2str(tensor_slice(-bias+center,0+center),'%.3f'))