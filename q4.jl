function simpleCountingSort(massive::AbstractArray) 
    acc = zeros(Int64, length(massive))
    for i in 1:length(massive)
        acc[massive[i]] = acc[massive[i]] + 1;
    end
    pos = 1;
    for number in 1:length(acc)
        for i in 1:acc[number]
            massive[pos] = number;
            pos = pos + 1;
        end
    end
end
# Нерабочий код: (хз вообще зачем он нужен, whatever)
#=
function countSort(massive::AbstractArray) # от 0 до 100
    acc = zeros(Int64, length(massive))

    for i in 1:length(massive)
        acc[i] += 1
    end
    j = 1
    for i in 1:length(massive)
        if acc[i] > 0
            for k in 1:acc[i]
                massive[j] = i
                j += 1
            end
        end
    end
end
=#
