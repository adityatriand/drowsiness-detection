$(document).ready(function() {
    var path = window.location.href;
    if(path.includes('realtime')){
        var base_url = window.location.origin;
        path = base_url + '/detection';
    }
     $('ul a').each(function() {
      if (this.href === path) {
       $(this).addClass('active');
      }

     });

     $('#realtime').click(function(){
        var base_url = window.location.origin;
        window.location.replace(base_url+'/detection/realtime');
     });

     $('#input').click(function(){
        var menu = `<div><h6 class="title-two">Pilih Gambar</h6><form action="" method="POST" enctype="multipart/form-data"><input class="form-control" type="file" name="face" accept="image/*" required /><button type="submit" class="btn btn-info mt-3 fw-bold">Prediksi</button></form></div>`;
        $('#menu').html(menu);
     });
});