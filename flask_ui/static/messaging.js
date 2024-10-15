const socket = io();
socket.on('files-changed', event => {
  $.ajax({
    url: "/score-ocr/files",
    type: "post",
    success: function(response) {
      $("#file-list").html(response);
    }
  });
});

function onclick_path(id) {
  $.ajax({
    url: "/score-ocr/load_data/" + id,
    type: "post",
    success: function(response) {
      $("#data").html(response);
    }
  });
}