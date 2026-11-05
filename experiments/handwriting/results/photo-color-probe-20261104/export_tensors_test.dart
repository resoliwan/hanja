import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:hanja_app/adapter/photo/photo_rasterizer.dart';
import 'package:hanja_app/adapter/photo/photo_crop.dart';
import 'package:hanja_app/adapter/recognition/ink_normalization.dart';
import 'package:hanja_app/dm/model/photo_selection.dart';
const directory='/Users/leo/aoo_workspace/hanja/experiments/handwriting/results/photo-color-probe-20261104';
int otsu(img.Image gray) {
 final hist=List<int>.filled(256,0);
 for(final p in gray) {hist[p.r.toInt()]++;}
 final total=gray.width*gray.height;
 var sum=0; for(var i=0;i<256;i++){sum+=i*hist[i];}
 var count=0,partial=0,threshold=0; var best=-1.0;
 for(var i=0;i<255;i++) {
 count+=hist[i]; partial+=i*hist[i];
 if(count==0||count==total)continue;
 final d=partial/count-(sum-partial)/(total-count);
 final variance=count*(total-count)*d*d;
 if(variance>best){best=variance;threshold=i;}
 }
 return threshold;
}
void main(){test('Export color comparison tensors',(){
 final manifest=jsonDecode(File('$directory/manifest.json').readAsStringSync()) as List;
 Directory('$directory/tensors').createSync();
 Directory('$directory/model-inputs').createSync();
 for(final row in manifest){
 final bytes=File(row['path'] as String).readAsBytesSync();
 final input=(bytes:bytes,selection:const PhotoSelection(0,0,1,1));
 final gray=img.grayscale(img.decodeJpg(cropPhotoSquare(input))!);
 final variants=<String,Float32List>{'current-binary':normalizePhotoInk(input),'gray-normalized':normalizeInk(gray.getBytes(order:img.ChannelOrder.rgba),gray.width,gray.height)};
 final resized=img.copyResize(gray,width:128,height:128,interpolation:img.Interpolation.cubic);
 variants['direct-gray']=Float32List.fromList([for(final p in resized)1-p.r/127.5]);
 final threshold=otsu(gray);
 var dark=0,edge=0;
 for(var y=0;y<gray.height;y++){for(var x=0;x<gray.width;x++){
 if(x==0||y==0||x==gray.width-1||y==gray.height-1){edge++;if(gray.getPixel(x,y).r<=threshold)dark++;}
 }}
 final invert=dark>edge/2;
 final binary=img.Image(width:gray.width,height:gray.height);
 for(final p in gray){final ink=(p.r<=threshold)!=invert;final v=ink?0:255;binary.setPixelRgb(p.x,p.y,v,v,v);}
 variants['auto-polarity-binary']=normalizeInk(binary.getBytes(order:img.ChannelOrder.rgba),gray.width,gray.height);
 for(final entry in variants.entries){
 final name='${row['id']}-${entry.key}';
 File('$directory/tensors/$name.f32').writeAsBytesSync(entry.value.buffer.asUint8List());
 final preview=img.Image(width:128,height:128);
 for(var y=0;y<128;y++){for(var x=0;x<128;x++){final v=((1-entry.value[y*128+x])*127.5).round().clamp(0,255);preview.setPixelRgb(x,y,v,v,v);}}
 File('$directory/model-inputs/$name.png').writeAsBytesSync(img.encodePng(preview));
 }
 }
});}
