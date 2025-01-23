def plot_and_accept(wave, num, y, cuff, patient_idx, plot_idx):
    fig, ax = plt.subplots(4, figsize=(15, 12))
    result = {'action': None}
    
    def on_key(event):
        if event.key == 'alt':
            result['action'] = "accept"
            plt.close(fig)
        elif event.key == 'a':
            result['action'] = "reject"
            plt.close(fig)
        elif event.key == 'f':
            result['action'] = "save"
            plt.close(fig)

    ax[0].plot(y * 6.5 + 81.2, label="ground truth MAP", c="darkred")
    ax[0].set_xlabel('Time (seconds)', size=18)
    ax[0].set_ylabel('MAP (mmHg)', size=18)
    #ax[0].set_ylim(50, 115)
    ax[0].set_title(f"Patient {str(patient_idx)} batch #{str(plot_idx)} MAP predictions ")
    ax[0].legend(["predicted MAP", "ground truth MAP"])

    ax[1].plot(num[:, 0], label="NRA", c="yellow")
    ax[1].plot(num[:, 1], label="HR", c="peru")
    ax[1].legend(["NRA", "HR"])
    ax[1].set_xlabel('Time (seconds)', size=18)
    ax[1].set_ylabel('Normalized feature (-) ', size=18)
    ax[1].set_title(f"{str(plot_idx)} features ")
    ax[2].plot(wave[10000:12000, 0], label="Pleth")
    ax[2].plot(wave[10000:12000, 1], label="ECG")
    ax[2].legend(["Pleth", "ECG"])
    ax[3].plot(wave[:, 0], label="Pleth")
    ax[3].plot(wave[:, 1], label="ECG")
    ax[3].legend(["Pleth", "ECG"])
    fig.suptitle('Press ENTER to accept or ESC to reject', fontsize=14)
    # Connect the key press event
    result_string = fig.canvas.mpl_connect('key_press_event', on_key)
    # Show plot and wait for input
    plt.show(block=True)
    return result["action"]