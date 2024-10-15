export function ajax(id_button, id_container, url){
    $(id_button).keyup(function(){
        const text = $(this).val();

        $.ajax({
            url: url,
            type: "post",
            data: {input: text},
            success: function(response) {
                $(id_container).html(response);
            },
            error: function(xhr) {
                //Do Something to handle error
                console.log(xhr);
            }
        });
    });
}